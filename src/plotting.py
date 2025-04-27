import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.widgets import Slider
import numpy as np

def plot_concentrations(concentr_diffs, title="", custom_labels=None, Hb_idxs=(0, 1)):
    num_molecules, num_spectra = concentr_diffs.shape
    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = custom_labels
    if custom_labels is None:
        labels = ["HbO2", "Hbb","diff oxyCCO","diff cc red","water","fat"]

    for i in range(num_molecules):
        if labels[i] == "water" or labels[i] == "fat":
                plt.plot(concentr_diffs[i, :]/10, label=labels[i]+"/10", linewidth=2)
        else:
            plt.plot(concentr_diffs[i, :], label=labels[i], linewidth=2)

    if Hb_idxs is not None:
        plt.plot(concentr_diffs[Hb_idxs[0], :] + concentr_diffs[Hb_idxs[1], :], color='#d62728', label="HbT",linewidth=2)
    plt.title(title)
    plt.legend()

#(tissue, model, params)
def plot_concentrations_bar(concentrations, tissue_labels, model_labels, param_labels, water_scale_factor=(1/100)):

    concentrations[:, :, -2:] *= water_scale_factor
    
    num_tissues, num_models, num_params = np.array(concentrations).shape
    for tissue_idx in range(num_tissues):
        x = np.arange(num_models)
        fig, ax = plt.subplots(layout="constrained")
        multiplier = 0
        width = 0.125
        for param_idx in range(num_params):
            offset = width * multiplier
            rects = ax.bar(x + offset, concentrations[tissue_idx, :, param_idx], width, label=param_labels[param_idx])
            #ax.bar_label(rects, padding=3)
            multiplier += 1
        
        ax.set_ylabel("Concentration (mM)")
        ax.set_xticks(x + width, model_labels)
        ax.legend()
        ax.set_title(tissue_labels[tissue_idx])

def plot_spectrum(spectrum, wavelengths=None, label="", title=""):
    plt.figure()
    plt.plot(wavelengths, spectrum, label=label)
    plt.title(title)
    plt.legend()

def plot_spectra(spectra, wavelengths=None, labels=None, title=""):
    if labels is None:
        labels = [""] * len(spectra)
    plt.figure()
    for i in range(len(spectra)):
        plt.plot(wavelengths, spectra[i], label=labels[i])
    plt.title(title)
    plt.legend()


# (molecules, wavelengths, spectra)
def plot_spectra_slider(spectra, wavelengths=None, labels=None, title=""):
    _, num_spectra_idxs = spectra[0].shape
    num_spectra = len(spectra)
    spectra = np.stack(spectra)
    idx = num_spectra_idxs // 2
    ymax = np.max(spectra[:, :, idx], axis=(0, 1)) 
    ymin = np.min(spectra[:, :, idx], axis=(0, 1))
    delta_y = 0.05*(ymax-ymin)/2
    if labels is None:
        labels = [""] * len(spectra)
    fig, ax = plt.subplots()
    ax.set_ylim(ymin=ymin - delta_y, ymax=ymax + delta_y)
    for i in range(num_spectra):
        ax.plot(wavelengths, spectra[i, :, idx], label=labels[i])
    fig.subplots_adjust(left=0.25, bottom=0.25)
    ax.legend(labels, loc="upper right")
    ax_slider = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Index",
        valmin=0,
        valmax=num_spectra_idxs,
        valinit=idx,
        valstep=np.arange(0, num_spectra_idxs),
        orientation="horizontal"
    )

    def update(val):
        idx = int(val)
        ax.cla()
        ymax = np.max(spectra[:, :, idx], axis=(0, 1)) 
        ymin = np.min(spectra[:, :, idx], axis=(0, 1))
        for i in range(num_spectra):
            ax.plot(wavelengths, spectra[i, :, idx], label=labels[i])
        ax.set_ylim(ymin=ymin - delta_y, ymax=ymax + delta_y)
        ax.legend(labels, loc="upper right")

    
    slider.on_changed(update)
    plt.title(title)

    return slider


def plot_concentration_imgs(c_imgs, reference_idx=None):

    num_molecules = c_imgs.shape[0]
    #labels = ["HbO2", "Hbb","diff oxyCCO","diff cc red","water","fat"]
    num_cols = (num_molecules+1)//2
    fig, axs = plt.subplots(nrows=2, ncols=num_cols)
    cmap = get_cmap("viridis")
    cmap.set_bad(alpha=0)

    for i in range(num_molecules):
        ax = axs[i//num_cols, i%num_cols]
        pos = ax.imshow(c_imgs[i])
        fig.colorbar(pos, ax=ax)
        #ax.set_title(labels[i])
        if reference_idx is not None:
            ax.plot(
                reference_idx[0],
                reference_idx[1],
                marker="o",
                markersize=10,
                markeredgecolor="red",
                markerfacecolor="None"
            )
    
    


