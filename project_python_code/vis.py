import matplotlib
from matplotlib import pyplot as plt


def plot_sr_results_np(x, y, z, titles=None, suptitle="", save=None):
    fig, ax = plt.subplots(1, 3, dpi=200)
    if titles is None:
        titles = ["","",""]
    for _i, (_a, _t) in enumerate(zip([x,y,z], titles)):
        print(_a.shape)
        ax[_i].imshow(_a)
        ax[_i].set_title(_t)
        ax[_i].axis('off')
    plt.set_cmap("magma")
    plt.suptitle(suptitle)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()



    
def plot_timeseries(data, idx,suptitle="" ):
    fig, ax = plt.subplots(1, len(idx), dpi=200)
    for _i, _ax in enumerate(ax):
        _ax.imshow(data[idx[_i],...])
        _ax.set_title(f"Slice {idx[_i]}")
        _ax.axis('off')
    plt.suptitle(suptitle)
    plt.set_cmap("magma")
    plt.tight_layout()
    plt.show()


def plot_timeseries_bilinear(data, idx, suptitle="", save=None):
    fig, ax = plt.subplots(1, len(idx), dpi=200)
    for _i, _ax in enumerate(ax):
        _ax.imshow(data[_i])
        _ax.set_title(f"Slice {idx[_i]}")
        _ax.axis('off')
    plt.suptitle(suptitle)
    plt.set_cmap("magma")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()



def plot_timeseries_temporal_dip(data, idx, suptitle="", save=None):
    fig, ax = plt.subplots(1, len(idx), dpi=200)
    for _i, _ax in enumerate(ax):
        _ax.imshow(data[_i].clone().detach().cpu().numpy()[0,0])
        _ax.set_title(f"Slice {idx[_i]}")
        _ax.axis('off')
    plt.suptitle(suptitle)
    plt.set_cmap("magma")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()



def plot_timeseries_torch(x_tensors, y_tensors, idx):
    fig, ax = plt.subplots(2, len(x_tensors))
    for i in range(len(x_tensors)):
        ax[0, i].imshow(x_tensors[i].detach().cpu().numpy()[0,0])
        ax[1, i].imshow(y_tensors[i].detach().cpu().numpy()[0,0])
        ax[0,i].set_title(f"SR Output {idx[i]}")
        ax[1,i].set_title(f"LR Input {idx[i]}")
    plt.set_cmap("magma")
    plt.tight_layout()
    plt.show()