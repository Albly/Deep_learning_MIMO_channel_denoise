from torch.nn.functional import mse_loss
import torch

def complex_MSE(x_real, x_hat):
    '''Mean squared error generalized for complex values'''

    assert x_real.shape == x_hat.shape, 'Sizes of both values must be the same, but got {0} and {1} instead'.format(x_real.shape, x_hat.shape)
    
    mse = torch.mean(torch.abs(x_real - x_hat)**2)
    return mse


def complex_LOG_MSE(x_real, x_hat):
    '''Mean squared error generalized for complex values'''

    assert x_real.shape == x_hat.shape, 'Sizes of both values must be the same, but got {0} and {1} instead'.format(x_real.shape, x_hat.shape)
    
    eps = torch.tensor([0.0000001])
    mlse = torch.mean(10*torch.log10(torch.abs(x_real-x_hat)**2 + eps))
    return mlse
