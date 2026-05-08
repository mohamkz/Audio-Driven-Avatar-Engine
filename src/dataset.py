import torch
from torch.utils.data import Dataset
import numpy as np

class LipSyncDataset(Dataset):
    def __init__(self, audio_path, lip_path, seq_len=30):
        self.audio_data = np.load(audio_path)
        self.lip_data = np.load(lip_path).reshape(len(np.load(lip_path)), -1)
        self.seq_len = seq_len
        self.step_size = self.audio_data.shape[0] // self.lip_data.shape[0]

    def __len__(self):
        return len(self.lip_data) - self.seq_len

    def __getitem__(self, idx):
        y = self.lip_data[idx : idx + self.seq_len]
        audio_start = idx * self.step_size
        audio_end = (idx + self.seq_len) * self.step_size
        x = self.audio_data[audio_start : audio_end]
        return torch.FloatTensor(x), torch.FloatTensor(y)