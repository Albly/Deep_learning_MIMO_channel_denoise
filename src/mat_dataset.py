import torch
import scipy.io
import mat73
from torch.utils.data import Dataset


def get_noise(signal, SNR = -16):
    signal_power = torch.mean(torch.abs(signal)**2)
    noise_power = signal_power * 10**(-SNR/10)
    noise_var = torch.sqrt(noise_power/2)
    noise =  noise_var*torch.randn(signal.shape) + 1j*noise_var*torch.randn(signal.shape) 
    return noise.type(torch.complex64)


class MAT_Dataset(Dataset):
    def __init__(self, path : str, 
                       UEs : list,
                       is_normalize : bool = False, 
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
        
        self.H = torch.permute(self.H, [3, 0 , 1, 2]).type(torch.complex64)
        #self.H = self.H * 2**8
        
    def __len__(self):
        return self.H.shape[0]


    def normalize_power(self, signal):
        signal_power = torch.mean(torch.abs(signal)**2)
        return signal/torch.sqrt(signal_power)


    def __getitem__(self, idx):
        data = self.H[idx]

        if self.is_normalize:
            data = self.normalize_power(data)
        
        return data  

class EVAL_Dateset(Dataset):
    def __init__(self, path: str,
                 method : str = 'scipy'):
        self.path = path

        if method == 'scipy':
            mat = scipy.io.loadmat(path)
        elif method == 'mat73':
            mat = mat73.loadmat(path)
        
        self.H = torch.from_numpy(mat['H']).type(torch.complex64)
        self.N = torch.from_numpy(mat['N']).type(torch.complex64)
        self.norma = []
    
    
    def __len__(self):
        return self.H.shape[0]
    

    def __getitem__(self, idx):
        data = self.H[idx]
        noise = self.N[idx]
        return data, noise

