import torch
from torch import nn
from src.preprocessing import AntFreq2BeamTime_3D, BeamDelay2AntFreq_3D, Pol2Dim, PowerNoiseNormalizer, CenterCut2
from src.plotting import project3d
from itertools import cycle
from src.plotting import project3d , display_losses 
from src.mat_dataset import get_noise


def train_model(model,cfg,epochs, trainLoader, testLoader, optimizer, scheduler, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    prev_val_loss = 10000000
    train_loss_history = []
    test_loss_history = []

    train_info_loss = []
    test_info_loss = []

    for epoch in range(epochs):
        for train_signal, test_signal in zip(trainLoader, cycle(testLoader)):
            train_noise = get_noise(signal = train_signal, SNR = -5)
            train_signal, train_noise  = train_signal.to(device), train_noise.to(device)

            optimizer.zero_grad()
            Power_noise = torch.sum(torch.abs(train_noise)**2 , dim = (1,2,3))
            sigma = Power_noise / (cfg['N_time']*cfg['N_Az']*cfg['N_El']*cfg['N_pol'])

            denoised_data, pred_noise, l1_loss = model(train_signal + train_noise, norma = torch.sqrt(sigma))
            noise_normed = model.left_transform(train_noise)

            loss = criterion(pred_noise, noise_normed)
            data_loss = criterion(denoised_data, train_signal)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_history.append(loss.item())
            train_info_loss.append(data_loss.item())

            model.eval()

            test_noise = get_noise(signal = test_signal, SNR = -5)
            test_signal, test_noise = test_signal.to(device), test_noise.to(device)

            with torch.no_grad():
                Power_noise = torch.sum(torch.abs(test_noise)**2 , dim = (1,2,3))
                sigma = Power_noise / (cfg['N_time']*cfg['N_Az']*cfg['N_El']*cfg['N_pol'])
                
                denoised_data, pred_noise,l1_loss = model(test_signal + test_noise, norma = torch.sqrt(sigma), plotimage = True)
                noise_normed = model.left_transform(test_noise)

                loss = criterion(pred_noise, noise_normed)
                data_loss = criterion(denoised_data, test_signal)

                test_loss_history.append(loss.item())
                test_info_loss.append(data_loss.item())

                project3d(model.Pol_dim(model.left_transform(test_signal), 'backward').cpu().detach())
                display_losses(train_loss_history, test_loss_history, train_info_loss, test_info_loss)



def weights_init_kaiming(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data, a = 0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            pass
            #nn.init.uniform(m.weight.data, 1e-4, 1e-3)
           #m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            #nn.init.constant(m.bias.data, 0.0)

class DnCNN3D(nn.Module):
    def __init__(self, 
                 channels, 
                 num_of_layers = 17,
                 num_features = 64, 
                 apply_kaiming = True):
        
        super(DnCNN3D, self).__init__()
        kernel_size = 3
        features = num_features
        layers = []
        layers.append(nn.Conv3d(in_channels=channels, out_channels=features, kernel_size=kernel_size,padding = 'same', bias = True, padding_mode='circular'))
        layers.append(nn.ReLU(inplace=True))
        for i in range(num_of_layers-2):
            dill = 3 if i%2 == 0 else 1
            layers.append(nn.Conv3d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding='same', bias=True, dilation = dill,  padding_mode='circular'))
            layers.append(nn.BatchNorm3d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding = 'same', bias=True, padding_mode='circular'))
        self.dncnn = nn.Sequential(*layers)

        if apply_kaiming:
            self.dncnn.apply(weights_init_kaiming)

    def forward(self, x):
        out = self.dncnn(x)
        return out     


class Beam_DnCNN_3D(torch.nn.Module):
    def __init__(self, cfg, n_layers = 20, n_features = 64):
        super(Beam_DnCNN_3D, self).__init__()

        self.Denoiser = DnCNN3D(cfg['N_UE_ANT'] * cfg['N_pol'] * 2, n_layers, n_features)
        self.BeamTransform = AntFreq2BeamTime_3D(n_hor = cfg['N_hor'],
                                                 n_ver = cfg['N_ver'],
                                                 n_pol = cfg['N_pol'],
                                                 out_size = (cfg['N_Az'], cfg['N_El'], cfg['N_time']),
                                                 )
        
        self.AntTransform = BeamDelay2AntFreq_3D(n_hor = cfg['N_hor'],
                                               n_ver = cfg['N_ver'], 
                                               n_pol = cfg['N_pol'], 
                                               n_freq = cfg['N_subc'],
                                               )
        
        self.WindowCut = CenterCut2(cfg['window_cut'])
        self.Normalizer = PowerNoiseNormalizer()
        self.Pol_dim = Pol2Dim()
        
    def left_transform(self, x, norma = None):
        with torch.no_grad():
            x = self.BeamTransform(x)
            x = self.Pol_dim(x, mode = 'forward')
            x = self.Normalizer(x, mode = 'norm', norma = norma)
            return x


    def right_transform(self, x):
        with torch.no_grad():
            x = self.Normalizer(x, mode = 'denorm', norma = None)
            x = self.Pol_dim(x, mode = 'backward')
            x = self.AntTransform(x)
            return x

    def dim_to_complex(self,x):
        n_dim = x.shape[1]
        assert n_dim % 2 == 0
        return (x[:, 0:n_dim//2, :, :] + 1j*x[:, n_dim//2:n_dim, :, :])# =============================torch.unsqueeze((x[:,0,:,:,:] + 1j*x[:,1,:,:,:]), dim = 1) 

    def complex_to_dim(self,x):
        return torch.concat((torch.real(x), torch.imag(x)), dim = 1)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, x, norma = None, plotimage = False):
        x = self.left_transform(x, norma)
        x_cat = self.complex_to_dim(x)
        
        with torch.no_grad():
            x_cat = self.WindowCut(x_cat)
        noise_pred = self.Denoiser(x_cat)
        
        noise_pred = self.WindowCut(noise_pred, mode = 'backward')
        noise_pred = self.dim_to_complex(noise_pred)
        
        x_pred = x - noise_pred
        x_denoised = self.right_transform(x_pred)

        if plotimage:
            project3d(self.Pol_dim(x_pred, 'backward').cpu().detach())


        l1_weight = 0.05
        #l1_parameters = []
        #for parameter in self.Denoiser.parameters():
        #    l1_parameters.append(parameter.view(-1))

        #L1 = l1_weight * self.compute_l1_loss(torch.cat(l1_parameters))
        L1 = l1_weight * torch.mean(abs(x_pred)) 

        return x_denoised, noise_pred, L1



