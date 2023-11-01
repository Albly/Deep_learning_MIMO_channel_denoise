import torch
import scipy.io
import mat73
from torch.utils.data import Dataset


# def get_noise(signal, SNR = -16):
#     signal_power = torch.mean(torch.abs(signal)**2)
#     noise_power = signal_power * 10**(-SNR/10)
#     noise_var = torch.sqrt(noise_power/2)
#     noise =  noise_var*torch.randn(signal.shape) + 1j*noise_var*torch.randn(signal.shape) 
#     return noise.type(torch.complex64)


def gen_noise(H, SNR):
    N_tti, N_ue_ant, N_bs, N_subc = H.shape[0], H.shape[1],H.shape[2], H.shape[3]
    noise_SRS = torch.sqrt(torch.tensor([0.5])) * (torch.rand(N_tti, N_ue_ant, N_bs, N_subc) + 1j*torch.rand(N_tti, N_ue_ant, N_bs, N_subc))
    gain = torch.sqrt(torch.mean(H * H.conj(), dim = -1).unsqueeze(-1).repeat(1,1,1,N_subc))
    noise_SRS_normed = 10**(-SNR/20) * gain * noise_SRS
    return noise_SRS_normed



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

