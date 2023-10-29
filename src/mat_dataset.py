import torch
import scipy.io
import mat73
from torch.utils.data import Dataset, DataLoader

class MAT_Dataset(Dataset):
    def __init__(self, path : str, 
                       UEs : list,
                       is_normalize : bool = True, 
                       method : str = 'scipy'):
        # dataset path
        self.imgs_path = path

        if method == 'scipy':
            mat = scipy.io.loadmat(path)
        elif method == 'mat73':
            mat = mat73.loadmat(path)

        self.UEs = UEs
        self.N_UE = len(self.UEs)
        self.H = torch.tensor([], dtype = torch.complex64)
        self.is_normalize = is_normalize

        for ue in UEs:
            H_tmp = torch.from_numpy(mat['Hfrq'][0, ue][0,0][0])
            self.H = torch.concat((self.H, H_tmp), dim = -1)
        
        self.H = torch.permute(self.H, [3, 0 , 1, 2])
        
        
    def __len__(self):
        return self.H.shape[0]


    def normalize_power(self, signal):
        signal_power = torch.mean(torch.abs(signal)**2)
        return signal/torch.sqrt(signal_power)


    def __getitem__(self, idx):
        data = self.H[idx]

        if self.is_normalize:
            data = self.normalize_power(data)
        
        data = torch.unsqueeze(data,0)
        
        return data  
