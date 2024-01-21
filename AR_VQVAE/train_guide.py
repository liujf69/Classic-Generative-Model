import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from misc import fixseed
from guide import GuideTransformer
from vqvae import setup_tokenizer, TemporalVertexCodec
from feeder import Feeder_Guide

def train_args():
    parser = argparse.ArgumentParser(
        description = "Train_Guide", add_help = True,
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    # Train
    parser.add_argument("--device", default = [0], type = list, help = "GPUs for training.")
    parser.add_argument("--seed", default = 0, type = int, help = "seed for initializing training.")
    parser.add_argument("--epoch", default = 100, type = int, help = "the number of epochs for training.")
    parser.add_argument("--work_dir", default = "./work_dir/24_0117_Guide/", type = str, help = "the dir of saving output")
    parser.add_argument("--batch_size", default = 8, type = int, help = "batch size")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "learning rate")
    parser.add_argument("--weight_decay", type = float, default = 0.1, help = "weight decay")

    # Guide
    parser.add_argument("--resume_pth", default = './work_dir/24_0115_epoch100/VQVAE_best.pt', type = str, help = "the checkpoint path of vqvae") # vqvae_weight_path
    parser.add_argument("--layers", default = 6, type = int, help = "the num of autoregressive transformer layers")
    parser.add_argument("--dim", default = 64, type = int, help = "the feature dims in aotoregressive transformer")
    parser.add_argument("--max_seq_length", default = 300, type = int, help = "the frame nums of data")
    args = parser.parse_args()
    return args

class ModelTrainer:
    def __init__(self, args, model: GuideTransformer, tokenizer: TemporalVertexCodec) -> None:
        self.model = model.cuda()
        self.tokenizer = tokenizer
        self.epoch = args.epoch
        self.device = args.device
        self.batch_size = args.batch_size

        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index = self.tokenizer.n_clusters + 1, label_smoothing = 0.1)  # 交叉熵损失
        self.optimizer = optim.AdamW(self.model.parameters(), lr = args.lr, betas = (0.9, 0.99), weight_decay = args.weight_decay) # 优化器

    def load_dataset(self) -> (DataLoader, DataLoader):
        train_loader = DataLoader(
            dataset = Feeder_Guide(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_train_samples.txt', label_name_path ='./dataset/label_name.txt'),
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 8,
            drop_last = False)
        
        val_loader = DataLoader(
            dataset = Feeder_Guide(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_val_samples.txt', label_name_path ='./dataset/label_name.txt'),
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 8,
            drop_last = False)
        return train_loader, val_loader
    
    # 准备自回归的输入tokens和目标tokens
    def _prepare_tokens(self, motions: torch.Tensor) -> (torch.Tensor):
        motions = motions.reshape(motions.shape[0], motions.shape[1], -1) # [B, T, N, C] -> [B, T, NC]
        B, T, _ = motions.shape
        # target_tokens和input_tokens错位一个，形成自回归
        target_tokens = self.tokenizer.predict(motions) # 调用VQVAE获取out_indices [B T residual_depth] 
        target_tokens = target_tokens.reshape(B, -1) # [B T*residual_depth] # 目标tokens是VQVAE输出的码本数值
        input_tokens = torch.cat([   
                torch.zeros((B, 1), dtype = target_tokens.dtype, device = target_tokens.device) + self.model.tokens, # 第一个token用self.model.tokens表示
                target_tokens[:, :-1] # [B T*residual_depth-1]
            ], axis = -1) # input_tokens = [B, T*residual_depth] # 输入tokens 
        # new_mask = torch.ones((B, 1, 1, T), dtype = torch.bool) # All True
        return input_tokens, target_tokens, motions.reshape((B, T, -1))
    
    # 计算生成motion和源motion的L2损失
    def cal_l2loss(self, a: torch.Tensor, b: torch.Tensor) -> float:
        loss = (a - b) ** 2 # L2 Loss
        loss = self.sum_flat(loss) # [B]
        mse_loss_val = loss / self.sum_flat(torch.ones(a.shape)) # divide the elem num of each sample
        return mse_loss_val.mean()
    def sum_flat(self, tensor):
        return tensor.sum(dim = list(range(1, len(tensor.shape)))) # cal each sample loss [B]
    
    # cal_acc
    def compute_accuracy(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        # logtis.shape: [B T*residual_depth code_dim] target.shape [B T*residual_depth]
        probs = torch.softmax(logits, dim=-1)
        _, cls_pred_index = torch.max(probs, dim=-1) # get pred codebook index [B T*residual_depth]
        acc = (cls_pred_index.flatten(0) == target.flatten(0)).reshape(cls_pred_index.shape)  # [B T*residual_depth] -> [B*T*residual_depth] -> [B T*residual_depth]
        acc = self.sum_flat(acc).detach().cpu() # cal the nums of True (pred correct) # [B]
        acc_val = acc / self.sum_flat(torch.ones(target.shape)) * 100 # [B]
        return acc_val.mean()

    def train(self, epoch_idx: int):
        total_celoss = []
        total_l2loss = []
        total_perplexity = []
        total_acc = []

        self.model.train()
        for batch_idx, (gt_motion, label_idx, label_names, audio) in enumerate(tqdm(self.train_loader, ncols=40)):
            with torch.no_grad():
                gt_motion = gt_motion.float().cuda(self.device[0]) # B T N C
                gt_motion = torch.cat((gt_motion, torch.zeros(gt_motion.shape[0], gt_motion.shape[1], gt_motion.shape[2], 1).cuda(self.device[0])), dim = -1) # B T N C
                label_idx = label_idx.long().cuda(self.device[0]) # B
                label_input_ids = label_names['input_ids'].squeeze(1).cuda(self.device[0]) # B 1 10
                label_atten_mask = label_names['attention_mask'].squeeze(1).cuda(self.device[0]) # B 1 10
                label_token_typeID = label_names['token_type_ids'].squeeze(1).cuda(self.device[0]) # B 1 10
                audio = audio.float().cuda(self.device[0]) # B 48000 2
                # 准备自回归的输入tokens和目标tokens tokens:[B, T*residual_depth] down_gt:[B T NC]
                input_tokens, target_tokens, gt_motion = self._prepare_tokens(gt_motion) # 调用stage1的Encoder
                B, T = input_tokens.shape[0], input_tokens.shape[1]
            
            self.optimizer.zero_grad()
            # forward text_condition: th.Tensor, audio_condition: th.Tensor
            logits = self.model(tokens = input_tokens, text_ids = label_input_ids, text_masks = label_atten_mask, Text_tokenType = label_token_typeID,
                                 audio_condition = audio, cond_drop_prob=0.20) # 调用自回归transformer [B, T*residual_depth, code_dim]
            # 将生成的logits和target_tokens计算交叉熵损失，实现自回归损失计算
            loss = self.ce_loss(logits.reshape((B * T, -1)), target_tokens.reshape((B * T)).long())
            # backward
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                pred_tokens = torch.argmax(logits, dim = -1).view(input_tokens.shape[0], -1, self.tokenizer.residual_depth) # get the pred_tokens [B T residual_depth]
                pred_motions = self.tokenizer.decode(pred_tokens).detach().cpu() # 调用VQVAE解码pose # [B T NC]
                l2_loss = self.cal_l2loss(gt_motion.cpu(), pred_motions) # cal l2loss between gt_motion and pred_motion
                acc = self.compute_accuracy(logits, target_tokens) # 计算预测codebook index的准确率
                
                total_celoss.append(loss.item())
                total_l2loss.append(l2_loss.item())
                total_perplexity.append(np.exp(loss.item()))
                total_acc.append(acc.item())

        print("Training epoch ", epoch_idx)
        print("Autoregressive Token ce_loss: ", np.mean(total_celoss), "\tl2_loss: ", np.mean(total_l2loss),
                "\tperplexity: ", np.mean(total_perplexity), "\tacc: ", np.mean(total_acc))
        print("Pause")

    def eval(self, epoch_idx: int):
        total_celoss = []
        total_l2loss = []
        total_perplexity = []
        total_acc = []

        self.model.eval()
        for batch_idx, (gt_motion, label_idx, label_names, audio) in enumerate(tqdm(self.val_loader, ncols=40)):
            with torch.no_grad():
                gt_motion = gt_motion.float().cuda(self.device[0]) # B T N C
                gt_motion = torch.cat((gt_motion, torch.zeros(gt_motion.shape[0], gt_motion.shape[1], gt_motion.shape[2], 1).cuda(self.device[0])), dim = -1) # B T N C
                label_idx = label_idx.long().cuda(self.device[0]) # B
                label_input_ids = label_names['input_ids'].squeeze(1).cuda(self.device[0]) # B 1 10
                label_atten_mask = label_names['attention_mask'].squeeze(1).cuda(self.device[0]) # B 1 10
                label_token_typeID = label_names['token_type_ids'].squeeze(1).cuda(self.device[0]) # B 1 10
                audio = audio.float().cuda(self.device[0]) # B 48000 2
                input_tokens, target_tokens, gt_motion = self._prepare_tokens(gt_motion)
                logits = self.model(tokens = input_tokens, text_ids = label_input_ids, text_masks = label_atten_mask, Text_tokenType = label_token_typeID,
                                    audio_condition = audio, cond_drop_prob=0.20)
                
                B, T = input_tokens.shape[0], input_tokens.shape[1]
                loss = self.ce_loss(logits.reshape((B * T, -1)), target_tokens.reshape((B * T)).long())
                pred_tokens = torch.argmax(logits, dim = -1).view(input_tokens.shape[0], -1, self.tokenizer.residual_depth)
                pred_motions = self.tokenizer.decode(pred_tokens).detach().cpu() # [B T N]
                l2_loss = self.cal_l2loss(gt_motion.cpu(), pred_motions)
                acc = self.compute_accuracy(logits, target_tokens)

                total_celoss.append(loss.item())
                total_l2loss.append(l2_loss.item())
                total_perplexity.append(np.exp(loss.item()))
                total_acc.append(acc.item())

        print("testing epoch ", epoch_idx)
        print("Autoregressive Token ce_loss: ", np.mean(total_celoss), "\tl2_loss: ", np.mean(total_l2loss),
                    "\tperplexity: ", np.mean(total_perplexity), "\tacc: ", np.mean(total_acc))

    def run(self):
        self.train_loader, self.val_loader = self.load_dataset() # load dataset
        for epoch_idx in range(self.epoch):
            self.train(epoch_idx)
            self.eval(epoch_idx)
            print("pause")

def main(args):
    fixseed(args.seed) # fix random seed
    tokenizer = setup_tokenizer(args.resume_pth) # return VQVAE
    args_path = os.path.join(args.work_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    # load Autoregressive transformer
    model = GuideTransformer(
        tokens = tokenizer.n_clusters,
        emb_len = 798 if args.max_seq_length == 240 else 1998,
        num_layers = args.layers,
        dim = args.dim)
    # init trainer
    trainer = ModelTrainer(args, model, tokenizer)
    trainer.run()
    print("pause")

if __name__ == "__main__": 
    args = train_args()
    os.makedirs(args.work_dir, exist_ok=True)
    main(args)
