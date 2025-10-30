import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

class LFPDataset(Dataset):
    def __init__(self, cfg, aligned_data, step_onset_texture, step_onset_LD, val_idx, validation=False):
        """
        Args:
            cfg: config file
            aligned_data: aligned lfp data size (num_trials, num_samples, num_channels)
            step_onset_texture: step onset texture label size (num_trials, num_classes)
            step_onset_LD: step onset light/dark label size (num_trials, 1)
            val_idx: validation index
            validation: if true, validation dataset is used
        """
        self.cfg = cfg
        self.validation = validation
        
        # calculate bin size for extracting data
        self.extract_bin_len = int(self.cfg.preprocess.bin_msec*self.cfg.dat.params.sampling_rate/self.cfg.dat.params.ds_ratio/1000)

        # calculate ttl interval of video frame
        self.frame_ttl_interval = int((self.cfg.dat.params.sampling_rate/self.cfg.dat.params.ds_ratio)*(1/self.cfg.dat.params.video_fps))
        
        self.n_shifts = self.cfg.preprocess.aug_shift*2+1 # number of shifts for augmentation
        self.len_shift = int(self.frame_ttl_interval/self.cfg.preprocess.aug_shift) # length of shift

        # transpose so that channels are in the first dimension
        aligned_data = np.transpose(aligned_data, (0, 2, 1))

        # concatenate labels
        label = np.vstack([step_onset_texture, step_onset_LD]).T

        if self.validation:
            self.data = aligned_data[val_idx, :, self.frame_ttl_interval:-1*self.frame_ttl_interval]
            self.label = label[val_idx]
            self.len_data = self.data.shape[0]
            # print data shape
            print("Validation data shape:", self.data.shape)
            print("Validation label shape:", self.label.shape)

        else:
            self.data = np.delete(aligned_data, val_idx, axis=0)
            self.label = np.delete(label, val_idx, axis=0)
            self.len_data = self.data.shape[0]*self.n_shifts
            # print data shape
            print("Training data shape:", self.data.shape)
            print("Training label shape:", self.label.shape)

        # normalize data
        if self.cfg.preprocess.normalize:
            self.data = self.calc_meanstd(self.data, axis=2)

        # numpy to tensor
        self.data = torch.from_numpy(self.data).type(torch.FloatTensor)
        self.label = torch.from_numpy(self.label).type(torch.FloatTensor)
            
    def calc_meanstd(self, data, axis=None):
        """Function that normalizes data by mean and std
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / std

    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if self.validation:
            return self.data[idx], self.label[idx]
        
        else:
            # get the index of the shift
            trial_idx = idx//self.n_shifts
            shift_idx = idx%self.n_shifts

            # extract data start_idx
            start_idx = self.len_shift*shift_idx

            # get the data
            data = self.data[trial_idx, :, start_idx:start_idx+self.extract_bin_len]
            label = self.label[trial_idx]

            return data, label


