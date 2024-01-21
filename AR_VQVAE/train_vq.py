import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

import vqvae
from misc import fixseed
from feeder import Feeder_VQVAE

def train_args():
    parser = argparse.ArgumentParser(
        description = "Train_VQVAE", add_help = True,
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    # Train
    parser.add_argument("--device", default = [0], type = list, help = "GPUs for training.")
    parser.add_argument("--seed", default = 0, type = int, help = "seed for initializing training.")
    parser.add_argument("--epoch", default = 100, type = int, help = "the number of epochs for training.")
    parser.add_argument("--work_dir", default = "./work_dir", type = str, help = "the dir of saving output")
    parser.add_argument("--commit", default = 0.02, type = float, help = "hyper-parameter for the commitment loss")
    parser.add_argument("--lr", default = 2e-4, type = float, help = "max learning rate, default 0.0002")
    parser.add_argument("--lr_scheduler", default = [300_000], nargs = "+", type = int, help = "learning rate schedule (iterations)")
    parser.add_argument("--gamma", default = 0.05, type = float, help = "learning rate decay")
    parser.add_argument("--weight_decay", default = 0.0, type = float, help = "weight decay")
    parser.add_argument("--batch_size", default = 6, type = int, help = "batch size")
    # VQVAE
    parser.add_argument("--nb_joints", default = 17, type = int, help = "the number of joints used by the data.")
    parser.add_argument("--output_emb_width", default = 512, type = int, help = "feature dimensions output by encoder in VQ-VAE.")
    parser.add_argument("--code_dim", default = 512, type = int, help = "dimension of codebook embedding. (num of embedding)")
    parser.add_argument("--depth", default = 4, type = int, help = "depth of the network. (num of VQ)")
    args = parser.parse_args()
    return args

class ModelTrainer:
    def __init__(self, vqvae_net: vqvae.TemporalVertexCodec, args: list[str]):
        self.device = args.device
        self.epoch = args.epoch
        self.work_dir = args.work_dir
        self.commit = args.commit 
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.global_step = 0

        self.vqvae_net = vqvae_net.cuda(self.device[0])
        self.optimizer = optim.AdamW(self.vqvae_net.parameters(), lr = self.lr,
            betas = (0.9, 0.99), weight_decay = args.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones = args.lr_scheduler, gamma = args.gamma)
        self.loss = torch.nn.SmoothL1Loss()
    
    def load_dataset(self) -> (DataLoader, DataLoader):
        train_loader = DataLoader(
            dataset = Feeder_VQVAE(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_train_samples.txt'),
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 8,
            drop_last = False)
        
        val_loader = DataLoader(
            dataset = Feeder_VQVAE(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_val_samples.txt'),
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 8,
            drop_last = False)
        return train_loader, val_loader
    
    # loss function
    def _l2_loss(self, motion_pred, motion_gt, mask = None):
        if mask is not None:
            return self._masked_l2(motion_pred, motion_gt, mask)
        else:
            return self.loss(motion_pred, motion_gt)     
    def _masked_l2(self, a, b, mask):
        loss = self._l2_loss(a, b)
        loss = self.sum_flat(loss * mask.float())
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = self.sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val
    def sum_flat(self, tensor):
        """Take the sum over all non-batch dimensions."""
        return tensor.sum(dim=list(range(1, len(tensor.shape))))
    # def _vel_loss(self, motion_pred, motion_gt):
    #     model_results_vel = motion_pred[..., :-1] - motion_pred[..., 1:]
    #     model_targets_vel = motion_gt[..., :-1] - motion_gt[..., 1:]
    #     return self.loss(model_results_vel, model_targets_vel)
    
    def train(self, epoch_idx):
        self.vqvae_net.train()
        total_loss = []
        total_loss_motion = []
        total_loss_commit = []
        total_perplexity = 0
        for batch_idx, (gt_motion) in enumerate(tqdm(self.train_loader, ncols=40)):
            self.global_step += 1 # count the step 
            # self.scheduler.step() # change lr in each iter
            gt_motion = gt_motion.float().cuda(self.device[0]) # B T N C
            gt_motion = torch.cat((gt_motion, torch.zeros(gt_motion.shape[0], gt_motion.shape[1], gt_motion.shape[2], 1).cuda(self.device[0])), dim = -1) # B T N C
            pred_motion, loss_commit, perplexity = self.vqvae_net(gt_motion, mask = None)
            total_perplexity += perplexity

            loss_motion = self._l2_loss(pred_motion, gt_motion).mean()
            loss = loss_motion + self.commit * loss_commit
            total_loss.append(loss.data.cpu().numpy())
            total_loss_motion.append(loss_motion.data.cpu().numpy())
            total_loss_commit.append(loss_commit.data.cpu().numpy())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        epoch_loss = np.mean(total_loss)
        print("train_epoch_idx: ", epoch_idx, "\tperplexity: ", total_perplexity.item(), "\ttrain_loss: ", epoch_loss, 
            "\tmotion_loss: ", np.mean(total_loss_motion), "\tcommit_loss: ", np.mean(total_loss_commit))
        print("-------------------")
            
    @torch.no_grad()
    def eval(self, epoch_idx):
        self.vqvae_net.eval()
        total_loss = []
        total_perplexity = 0
        for batch_idx, (gt_motion) in enumerate(tqdm(self.val_loader, ncols=40)): 
            gt_motion = gt_motion.float().cuda(self.device[0]) # B T N C
            gt_motion = torch.cat((gt_motion, torch.zeros(gt_motion.shape[0], gt_motion.shape[1], gt_motion.shape[2], 1).cuda(self.device[0])), dim = -1) # B T N C
            pred_motion, _, perplexity = self.vqvae_net(gt_motion, mask = None)
            total_perplexity += perplexity
            loss_motion = self._l2_loss(pred_motion, gt_motion).mean()
            total_loss.append(loss_motion.data.cpu().numpy())

        epoch_loss = np.mean(total_loss)
        print("eval_epoch_idx: ", epoch_idx, "\tperplexity: ", total_perplexity.item(), "\teval_loss: ", epoch_loss) 
        print("-------------------")
        # best
        if(epoch_loss < self.best_loss):
            self.best_loss = epoch_loss
            torch.save(self.vqvae_net.state_dict(),  self.work_dir + '/VQVAE_best' + '.pt')
            print("best_epoch_idx: ", epoch_idx, "\tsave_best_weight!")
    
    def test(self, gt_motion:torch.Tensor, weight_path:str) -> torch.Tensor:
        weights = torch.load(weight_path)
        self.vqvae_net.load_state_dict(weights)
        self.vqvae_net.eval()
        pred_motion, _, _ = self.vqvae_net(gt_motion, mask = None)
        return pred_motion
        
    def run(self):
        self.train_loader, self.val_loader = self.load_dataset()
        self.best_loss = np.inf
        os.makedirs(self.work_dir, exist_ok = True)
        for epoch_idx in range(self.epoch):
            self.train(epoch_idx)
            self.eval(epoch_idx)
            # save model weights
            if (epoch_idx + 1 > 0) and (epoch_idx % 5 == 0):
                torch.save(self.vqvae_net.state_dict(),  self.work_dir + '/VQVAE_epoch' + str(epoch_idx) + '.pt')
            
def main(args):
    fixseed(args.seed)
    vqvae_net = vqvae.TemporalVertexCodec(
        n_vertices = args.nb_joints,
        latent_dim = args.output_emb_width,
        categories = args.code_dim,
        residual_depth = args.depth)
    
    # save training args
    args_path = os.path.join(args.work_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    trainer = ModelTrainer(vqvae_net, args)
    trainer.run()

if __name__ == "__main__":
    args = train_args()
    main(args)
    print("All done!")