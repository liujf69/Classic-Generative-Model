import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset

from transformers import BertTokenizer

class Feeder_VQVAE(Dataset):
    def __init__(self, data_root_path:str ='./dataset/origin_xy', data_samples_path:str ='./dataset/new_train_samples.txt'):
        super(Feeder_VQVAE, self).__init__()
        self.root_path = data_root_path
        self.data_samples_path = data_samples_path
        self.load_data()
    
    def load_data(self):
        self.data_samples = np.loadtxt(self.data_samples_path, dtype = str)
    
    def __len__(self): 
        return len(self.data_samples)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.root_path, self.data_samples[index]) + '.npy'
        data = np.load(data_path, allow_pickle = True)
        # ensure have the same frame_nums
        data_frame_pad = torch.zeros(300, data.shape[1], data.shape[2], data.shape[3])
        data_frame_pad[:data.shape[0]] = torch.from_numpy(data) 
        data = data_frame_pad.squeeze(1)
        return data

class Feeder_Guide(Dataset):
    def __init__(self, data_root_path:str ='./dataset/origin_xy', data_samples_path:str ='./dataset/new_train_samples.txt', label_name_path:str ='./dataset/label_name.txt'):
        super(Feeder_Guide, self).__init__()
        self.root_path = data_root_path
        self.data_samples_path = data_samples_path
        self.label_name_path = label_name_path
        self.Tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.load_data()

    def load_data(self):
        self.data_samples = np.loadtxt(self.data_samples_path, dtype = str)
        self.label_name = open(self.label_name_path, "r")
        self.label_name = self.label_name.readlines()

    def __len__(self): 
        return len(self.data_samples)
    
    def __getitem__(self, index) -> (torch.Tensor, int , str):
        # pose
        data_path = os.path.join(self.root_path, self.data_samples[index]) + '.npy'
        data = np.load(data_path, allow_pickle = True)
        # ensure have the same frame_nums
        data_frame_pad = torch.zeros(300, data.shape[1], data.shape[2], data.shape[3])
        data_frame_pad[:data.shape[0]] = torch.from_numpy(data) 
        data = data_frame_pad.squeeze(1)
        label_idx = int(self.data_samples[index][-3:])
        label_name = self.label_name[label_idx].strip()
        label_name = self.Tokenizer.encode_plus(text = (label_name), truncation = True, padding = 'max_length',
                                           add_special_tokens = True, max_length = 10, return_tensors = 'pt', return_attention_mask = True)
                                           

        # audio
        audio, _ = torchaudio.load('./dataset/scene01_audio.wav')
        audio = audio.T
        length = 300
        audio_per_frame = 1600
        start = np.random.randint(0, audio.shape[0] // 1600 - length)
        audio = audio[start * audio_per_frame : (start + length) * audio_per_frame, :] # [48000, 2]
        return data, label_idx, label_name, audio
    

if __name__ == "__main__":
    # Debug
    # train_loader = torch.utils.data.DataLoader(
    #             dataset = Feeder_VQVAE(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_train_samples.txt'),
    #             batch_size = 4,
    #             shuffle = True,
    #             num_workers = 4,
    #             drop_last = False)
    
    # val_loader = torch.utils.data.DataLoader(
    #         dataset = Feeder_VQVAE(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_val_samples.txt'),
    #         batch_size = 4,
    #         shuffle = False,
    #         num_workers = 4,
    #         drop_last = False)
    
    # for batch_size, (data) in enumerate(train_loader):
    #     data = data.float() # B T 17 2
    #     print("pasue")

    # Debug
    train_loader = torch.utils.data.DataLoader(
                dataset = Feeder_Guide(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_train_samples.txt', label_name_path = './dataset/label_name.txt'),
                batch_size = 4,
                shuffle = True,
                num_workers = 4,
                # collate_fn = collate_guide,
                drop_last = False)
    
    val_loader = torch.utils.data.DataLoader(
            dataset = Feeder_Guide(data_root_path = './dataset/origin_xy', data_samples_path = './dataset/new_val_samples.txt', label_name_path = './dataset/label_name.txt'),
            batch_size = 4,
            shuffle = False,
            num_workers = 4,
            # collate_fn = collate_guide,
            drop_last = False)
    
    for batch_size, (data, labels, names, audios) in enumerate(train_loader):
        data = data.float() # B T 17 2
        labels = labels.long() # B
        print("pasue") # B