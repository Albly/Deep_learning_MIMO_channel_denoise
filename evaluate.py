import torch
import os
import glob 

from src.mat_dataset import EVAL_Dateset
from torch.utils.data import DataLoader
from src.models import Beam_DnCNN_3D
import json
import scipy


if __name__ == '__main__':
    BATCH_SIZE = 5

    file_to_process = glob.glob('./data/*.mat')
    model_ckpt_path = glob.glob('./checkpoints/*.pt')

    dataset = EVAL_Dateset(file_to_process[0])
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle = False)
    cfg = json.load(open("model_config.txt"))

    model = Beam_DnCNN_3D(cfg = cfg, n_layers = 15, n_features = 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = torch.load(model_ckpt_path[0], map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    out = torch.tensor([], dtype = torch.complex64)

    

    with torch.no_grad():
        for signal, noise in dataloader:
            signal, noise = signal.to(device), noise.to(device)

            Power_noise = torch.sum(torch.abs(noise)**2 , dim = (1,2,3))
            sigma = Power_noise / (cfg['N_time']*cfg['N_Az']*cfg['N_El']*cfg['N_pol'])

            denoised_data, _, _ = model(signal + noise, norma = torch.sqrt(sigma))
            out = torch.concat((out, denoised_data ), dim = 0)

    out = out.to('cpu')

    scipy.io.savemat("./data/denoised.mat", {"H_denoised": out.numpy()})

