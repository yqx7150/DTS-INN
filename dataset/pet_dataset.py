import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class PetDataset(Dataset):
    def __init__(self, root_folder):
        self.hybrid_files = sorted(os.listdir(os.path.join(root_folder, 'hybrid')))
        self.fdg_files = sorted(os.listdir(os.path.join(root_folder, 'fdg')))
        self.fmz_files = sorted(os.listdir(os.path.join(root_folder, 'fmz')))

        self.root_folder = root_folder

    def __len__(self):
        return len(self.hybrid_files)

    def __getitem__(self, idx):
        hybrid_data = loadmat(os.path.join(self.root_folder, 'hybrid', self.hybrid_files[idx]))['data']
        fdg_data = loadmat(os.path.join(self.root_folder, 'fdg', self.fdg_files[idx]))['data']
        fmz_data = loadmat(os.path.join(self.root_folder, 'fmz', self.fmz_files[idx]))['data']
        # 在这里你可以对数据进行进一步的预处理，例如归一化等
        return hybrid_data, fdg_data, fmz_data


