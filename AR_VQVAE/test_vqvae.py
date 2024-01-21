import sys
sys.path.append('/apdcephfs_cq10/share_1567347/jinfullliu/NTU120_action/')
import vqvae
from train_vq_ljf import ModelTrainer

import torch
import argparse
import numpy as np
import cv2

def train_args():
    parser = argparse.ArgumentParser(
        description = "Train_VQVAE", add_help = True,
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    # Train
    parser.add_argument("--device", default = [0], type = list, help = "GPUs for training.")
    parser.add_argument("--seed", default = 0, type = int, help = "seed for initializing training.")
    parser.add_argument("--epoch", default = 10, type = int, help = "the number of epochs for training.")
    parser.add_argument("--work_dir", default = "./work_dir", type = str, help = "the dir of saving output")
    parser.add_argument("--commit", default = 0.02, type = float, help = "hyper-parameter for the commitment loss")
    parser.add_argument("--lr", default = 2e-4, type = float, help = "max learning rate, default 0.0002")
    parser.add_argument("--lr_scheduler", default = [300_000], nargs = "+", type = int, help = "learning rate schedule (iterations)")
    parser.add_argument("--gamma", default = 0.05, type = float, help = "learning rate decay")
    parser.add_argument("--weight_decay", default = 0.0, type = float, help = "weight decay")
    parser.add_argument("--batch_size", default = 4, type = int, help = "batch size")
    # VQVAE
    parser.add_argument("--nb_joints", default = 17, type = int, help = "the number of joints used by the data.")
    parser.add_argument("--output_emb_width", default = 512, type = int, help = "feature dimensions output by encoder in VQ-VAE.")
    parser.add_argument("--code_dim", default = 512, type = int, help = "dimension of codebook embedding. (num of embedding)")
    parser.add_argument("--depth", default = 4, type = int, help = "depth of the network. (num of VQ)")
    args = parser.parse_args()
    return args

def show(pose, name):
    img = np.zeros((1080, 1920, 3), dtype = np.uint8)
    frame = pose[0, 0, :, :-1] # 17 2
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
        [7, 9], [8, 10], [9, 11], [1, 2], [1, 3], [2, 4], [3, 5], [1, 7], [1, 6]
    ]
    for idx in skeleton:
        st_x = int(frame[idx[0] - 1][0])
        st_y = int(frame[idx[0] - 1][1])
        ed_x = int(frame[idx[1] - 1][0])
        ed_y = int(frame[idx[1] - 1][1])
        cv2.line(img, (st_x, st_y), (ed_x, ed_y), (0, 255, 0), 2)
    cv2.imwrite('./NTU120_action/' + name, img)

if __name__ == "__main__":
    args = train_args()
    weight_path = './NTU120_action/work_dir/24_0115_epoch100/VQVAE_epoch50.pt'
    test_pose_path = './NTU120_action/dataset/origin_xy/S001C001P001R001A001.npy'
    test_pose = np.load(test_pose_path, allow_pickle = True)
    test_pose = torch.cat((torch.from_numpy(test_pose), torch.zeros(test_pose.shape[0], test_pose.shape[1], test_pose.shape[2], 1)), dim = -1)
    vqvae_net = vqvae.TemporalVertexCodec(
        n_vertices = args.nb_joints,
        latent_dim = args.output_emb_width,
        categories = args.code_dim,
        residual_depth = args.depth)
    
    trainer = ModelTrainer(vqvae_net, args)
    reconstruct_pose = trainer.test(test_pose.float().cuda(args.device[0]), weight_path)
    show(test_pose.data.cpu().numpy(), 'origin.png')
    show(reconstruct_pose.data.cpu().numpy(), 'recon.png')
    print("All done!")