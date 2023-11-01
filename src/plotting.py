from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pll
import torch
from IPython import display
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def plot4d(data, th = 0.0):
    data = data.numpy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data > th
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)

    cmap = pll.cm.jet
    m_cmap = cmap(np.arange(cmap.N))
    m_cmap[:,-1] = np.linspace(0,1, cmap.N)
    m_cmap = ListedColormap(m_cmap)

    sc = ax.scatter(x, y, z, c=data.flatten(), s=30.0 * mask, edgecolor="face", marker="o", cmap=m_cmap, linewidth=0)
    plt.title('Spatial spectrum', fontsize = 15)
    plt.xlabel('Azimuth', fontsize = 14)
    plt.ylabel('Elevation', fontsize = 14)
    ax.set_zlabel('delay', fontsize = 14)
    plt.tight_layout()

def plot4d_3x(data1, data2, data3, th = (0.0, 0.0, 0.0), titles= ('Spatial spectrum','Spatial spectrum','Spatial spectrum')):
    data1, data2, data3 = data1.numpy(), data2.numpy(), data3.numpy()

    fig = plt.figure(figsize=(22,7))

    for k, data in enumerate([data1, data2, data3]):
        ax = fig.add_subplot(1,3,k+1, projection="3d")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        mask = data > th[k]
        idx = np.arange(int(np.prod(data.shape)))
        x, y, z = np.unravel_index(idx, data.shape)
        cmap = pll.cm.jet
        m_cmap = cmap(np.arange(cmap.N))
        m_cmap[:,-1] = np.linspace(0,1, cmap.N)
        m_cmap = ListedColormap(m_cmap)
        sc = ax.scatter(x, y, z, c=data.flatten(), s=30.0 * mask, edgecolor="face", marker="o", cmap=m_cmap, linewidth=0)
        plt.title(titles[k], fontsize = 15)
        plt.xlabel('Azimuth', fontsize = 14)
        plt.ylabel('Elevation', fontsize = 14)
        ax.set_zlabel('delay', fontsize = 14)
        plt.tight_layout()


def project3d(x):
    PAD = 0.25
    s = torch.abs(x[0,0,:,:,:,:])
    s = torch.roll(s, shifts = 0, dims = -1)
    plt.figure(figsize = (19,13))
    am_az = torch.mean(s,dim = 0)
    am_el = torch.mean(s, dim = 1)
    am_time = torch.mean(s, dim = 3)
    plt.subplot(3,2,1); plt.imshow(am_az[:,0,:], cmap = 'jet'); plt.xlabel('Time'); plt.ylabel('Elevation'); plt.gca().set_aspect('auto'); plt.xlim([450,650]);
    plt.colorbar(orientation="horizontal", pad=PAD); plt.title('1-pol') 
    plt.subplot(3,2,2); plt.imshow(am_az[:,1,:], cmap = 'jet'); plt.xlabel('Time'); plt.ylabel('Elevation');plt.gca().set_aspect('auto'); plt.xlim([450,650]);
    plt.colorbar(orientation="horizontal", pad=PAD); plt.title('2-pol');
    plt.subplot(3,2,3); plt.imshow(am_el[:,0,:], cmap = 'jet'); plt.xlabel('Time'); plt.gca().set_aspect('auto'); plt.xlim([450,650]);
    plt.ylabel('Azimuth'); plt.colorbar(orientation="horizontal", pad=PAD); plt.title('1-pol')
    plt.subplot(3,2,4); plt.imshow(am_el[:,1,:], cmap = 'jet'); plt.xlabel('Time'); plt.ylabel('Azimuth'); plt.gca().set_aspect('auto'); plt.xlim([400,650]);
    plt.colorbar(orientation="horizontal", pad=PAD); plt.title('2-pol')
    plt.subplot(3,2,5); plt.imshow(am_time[:,:,0].T, cmap = 'jet'); plt.xlabel('Azimuth'); plt.ylabel('Elevation'); plt.gca().set_aspect('auto')
    plt.colorbar(orientation="horizontal", pad=PAD); plt.title('1-pol')
    plt.subplot(3,2,6); plt.imshow(am_time[:,:,1].T, cmap = 'jet'); plt.xlabel('Azimuth'); plt.ylabel('Elevation');plt.gca().set_aspect('auto')
    plt.colorbar(orientation="horizontal", pad=PAD); plt.title('2-pol')
    #plt.tight_layout()

def plot4d(data, th = 0.0):
    data = data.numpy()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data > th
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)

    cmap = pll.cm.jet
    m_cmap = cmap(np.arange(cmap.N))
    m_cmap[:,-1] = np.linspace(0,1, cmap.N)
    m_cmap = ListedColormap(m_cmap)

    sc = ax.scatter(x, y, z, c=data.flatten(), s=30.0 * mask, edgecolor="face", marker="o", cmap=m_cmap, linewidth=0)
    plt.title('Spatial spectrum of noise', fontsize = 16)
    plt.xlabel('Azimuth', fontsize = 15)
    plt.ylabel('Elevation', fontsize = 15)
    ax.set_zlabel('delay', fontsize = 15)
    plt.tick_params(axis='both', labelsize=13)
    plt.tight_layout()

def display_losses(train_loss_hist, test_loss_hist, train_info_loss, test_info_loss):
    display.clear_output(wait=True)
    plt.figure(figsize = (18,6))
    plt.subplot(1,2,1)
    plt.title("training loss")
    plt.xlabel("#iteration")
    plt.semilogy(train_loss_hist, 'b', label='train')
    plt.semilogy(test_loss_hist, 'r', label ='test')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.title('Signal loss')
    plt.xlabel('iteration')
    plt.ylabel('MSE')
    plt.semilogy(train_info_loss, 'b', label = 'train')
    plt.semilogy(test_info_loss, 'r', label = 'test')
    plt.legend()
    plt.show();