import torch
from torch.fft import fft, ifft, fftshift, ifftshift


def unsqueeze_n(x, n, dim):
    for i in range(n):
        x = torch.unsqueeze(x, dim = dim)
    return x

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class AntFreq2BeamTime_3D(torch.nn.Module):
    '''
        Transform antenna and frequency space into beam and delay using 
        Fourier transform
       [N_batch x N_channel x N_ants x N_subc] --> reshaping into 
       [N_batch x N_channels x n_hor x n_ver x n_pol x n_subc] --> FFT 
       [N_batch x N_channels x n_azimuth x n_elevation x n_pol x n_delay]
        '''
    def __init__(self, n_hor = 8, n_ver = 4, n_pol = 2, out_size = (64,32,256)):
        super(AntFreq2BeamTime_3D, self).__init__()
        self.n_hor = n_hor
        self.n_ver = n_ver
        self.n_pol = n_pol
        self.out_size = out_size

    def forward(self, x):
        assert len(x.shape) == 4
        n_batch, n_chan, n_subc = x.shape[0], x.shape[1], x.shape[3]
        x = torch.reshape(x, (n_batch, n_chan, self.n_hor, self.n_ver, self.n_pol, n_subc))

        x_a_t = ifft(x , n = self.out_size[2], dim = 5, norm = 'ortho')             # to time-domain
        x_a_t = ifftshift(x_a_t, dim = 5)
        
        x_b_t = fft(x_a_t, n = self.out_size[0], dim = 2, norm = 'ortho')
        x_b_t = fftshift(x_b_t, dim = 2)
        
        x_c_t = fft(x_b_t, n= self.out_size[1], dim = 3, norm ='ortho')
        x_c_t = fftshift(x_c_t, dim = 3)                  # shift beams
        
        return x_c_t
    

class BeamDelay2AntFreq_3D(torch.nn.Module):
    def __init__(self, n_hor = 8, n_ver = 4, n_pol = 2, n_freq = 48):
        super(BeamDelay2AntFreq_3D, self).__init__()
        #self.shifts = shifts 
        self.n_hor = n_hor 
        self.n_ver = n_ver 
        self.n_pol = n_pol
        self.n_freq = n_freq
    
    def forward(self, x):
        assert len(x.shape) == 6
        n_b , n_c , n_h, n_v , n_pol, n_time = x.shape

        x_c_t = ifftshift(x, dim = 3)
        x_b_t = ifft(x_c_t, n = n_v, dim = 3, norm = 'ortho')
        x_b_t = x_b_t[:,:,:,:self.n_ver,:,:]
        
        x_a_t = ifftshift(x_b_t, dim = 2)
        x_a_t = ifft(x_a_t, n = n_h, dim = 2, norm = 'ortho')
        x_a_t = x_a_t[:,:,:self.n_hor,:,:,:]
        
        x_a_f = fftshift(x_a_t, dim = 5)
        x_a_f = fft(x_a_f, n = n_time, dim = 5 , norm = 'ortho')
        x_a_f = x_a_f[:,:,:,:,:,:self.n_freq]

        out = torch.reshape(x_a_f , (n_b, n_c, self.n_hor * self.n_ver * self.n_pol, self.n_freq))

        return out 


class Pol2Dim(torch.nn.Module):
    def __init__(self):
        super(Pol2Dim, self).__init__()

    def forward(self, x, mode = 'forward'):
        if mode == 'forward':
            assert len(x.shape) == 6
            n_b, n_c, n_h, n_v, n_p, n_t = x.shape
            self.n_p, self.n_c = n_p, n_c

            x = torch.permute(x, (0,1,4,2,3,5))
            x = torch.reshape(x, (n_b, n_c * n_p, n_h, n_v, n_t))
            return x

        elif mode == 'backward':
            assert len(x.shape) == 5
            n_b, n_c, n_h, n_v, n_t = x.shape

            x = torch.reshape(x, (n_b, self.n_c, self.n_p, n_h, n_v, n_t))
            x = torch.permute(x, (0,1,3,4,2,5))
            return x


class PowerNoiseNormalizer(torch.nn.Module):
    def __init__(self):
        super(PowerNoiseNormalizer, self).__init__()
        self.norma = None
    
    def forward(self, x, mode ='norm', norma = None):
        if mode == 'norm':
            if norma != None:
                self.norma = unsqueeze_n(norma, n = 4, dim = 1)
            if self.norma == None:
                raise ValueError('No info about norm. Calling norm with param is required')
            return x/self.norma

        elif mode == 'denorm':
            if self.norma == None:
                raise ValueError('No info about norm. Calling norm is required at first')

            return x*self.norma

        else:
            raise ValueError('Only norm and denorm types are allowed')
        

class CenterCut2(torch.nn.Module):
    def __init__(self, radius = (None,None,None)):
        super(CenterCut2, self).__init__()
        self.r = radius
        self.background = None
        self.s_d, self.s_h, self.s_w = None, None, None
        self.D_slice, self.H_slice, self.W_slice  = None, None, None

    def forward(self,x, mode = 'forward'):
        if mode == 'forward':
            x_b, x_c ,x_d, x_h, x_w = x.shape               # n_batch, n_chan, n_depth, n_height and n_width

            x_dc, x_hc, x_wc = x_d//2, x_h//2, x_w//2       # centers of dimentions

            #Powers = torch.reshape(x, (x_b, x_c//2, 2, x_d, x_h, x_w))
            Powers = torch.sum(x**2, 1)  

            Ps = torch.reshape(Powers, (x_b, -1))
            d,h,w = unravel_index(torch.argmax(Ps, dim = 1), (x_d, x_h, x_w))
            
            self.s_d = x_dc*torch.ones_like(d) - d                      # shifts 
            self.s_h = x_hc*torch.ones_like(h) - h
            self.s_w = x_wc*torch.ones_like(w) - w
            
            for i in range(x_b):
                x[i] = torch.roll(x[i], shifts = (self.s_d[i], self.s_h[i], self.s_w[i]), dims = (-3,-2,-1))
            
            self.D_slice = slice(x_dc-self.r[0] , x_dc+self.r[0], 1)
            self.H_slice = slice(x_hc-self.r[1] , x_hc+self.r[1], 1)
            self.W_slice = slice(x_wc-self.r[2] , x_wc+self.r[2], 1)
            
            self.background = torch.clone(x)

            return x[:,:,self.D_slice, self.H_slice, self.W_slice]

        elif mode == 'backward':
            if self.background == None:
                raise ValueError('No info. Forward calling is required before')
            
            out = self.background
            out[:,:,self.D_slice, self.H_slice, self.W_slice] = x#torch.clone(x) 

            for i in range(x.shape[0]):
                out[i] = torch.roll(out[i], shifts = (-self.s_d[i], -self.s_h[i], -self.s_w[i]), dims = (-3,-2,-1))
            
            return out              

