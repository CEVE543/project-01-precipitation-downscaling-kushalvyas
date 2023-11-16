from matplotlib import pyplot as plt
import skimage.transform
import numpy as np
import cv2

def tensor2im(x):
    n,c, h,w =  x.shape
    x_np = x.clone().detach().cpu().numpy()[0].transpose(1,2,0) # h,w,c
    if x_np.shape[-1] > 3:
        return x_np.mean(-1)
    elif x_np.shape[-1] == 1:
        return x_np[...,0]
    return x_np
        

def plot_sr_results(inp, hr, lr, lr_res, hr_res):
    plt.figure()
    plt.subplot(131)
    plt.imshow(tensor2im(inp))
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(tensor2im(hr))
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(tensor2im(lr))
    plt.colorbar()
    plt.tight_layout()
    plt.set_cmap("magma")
    plt.show()


def plot_sr_results_np(x, y, z, title=None):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, dpi=200)
    if title is None:
        title = ["","",""]
    for _i, (_a, _t) in enumerate(zip([x,y,z], title)):
        ax[_i].imshow(_a)
        ax[_i].set_title(_t)
    plt.set_cmap("magma")
    plt.show()


def plot_sr(hr, lr):
    plt.figure(dpi=200)
    plt.subplot(121)
    plt.imshow(tensor2im(hr))
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(tensor2im(lr))
    plt.colorbar()
    plt.tight_layout()
    plt.set_cmap("magma")
    plt.show()


def get_baseline_bilinear(x, size):
    return cv2.resize(x, size, interpolation=cv2.INTER_LINEAR_EXACT)


def normalize(x):
    return (x - x.min())/(x.max() - x.min())

def step_normalize(x):
    pass

def save_single_image(fname, arr, cmap='magma',dpi=200):
    plt.figure(dpi=dpi)
    plt.imshow(arr)
    plt.axis('off')
    plt.set_cmap(cmap)
    plt.savefig(fname)
    # plt.show()