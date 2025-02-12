import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline
from eventwarping.window import LinearScalingWindow
from pathlib import Path

from eventwarping.eventseries import EventSeries

plots_folder = Path(__file__) / "plots"

def add_grid(ax):
    ax.grid(axis='y', linewidth=2, color='k')
    for x in range(27):
        ax.axvline(x-0.5, color='gray', linewidth=0.5)
    for y in range(48):
        ax.axhline(y - 1.5, color='gray', linewidth=0.5)
    return ax

def plot_warping_test_data(model, labels, label, name):
    """
    Plot the density before and after warping.
    """
    ticks = np.array([1.5,11.5,21.5,31.5, 41.5, 45.5]) - 2
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax    ', 'other']

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,6))

    ax = axs[0]
    im = model.series[labels == label].mean(axis=0).T
    implot = ax.imshow(im[5:49], vmin=0, vmax=0.65)
    ax.set_xlabel('t')
    ax.set_ylabel('item')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_title('before warping')

    ax = axs[1]
    im = model.get_counts(ignore_merged=True, filter_series=labels == label) / sum(labels==label)
    implot = ax.imshow(im[5:49], vmin=0, vmax=0.65)
    ax.set_xlabel('t')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_title('after warping')

    fig.colorbar(implot, location='right', ax=axs, shrink=0.75)

    fig.savefig(plots_folder / name, bbox_inches='tight')
    plt.close(fig)

def plot_density(models, name):
    """
    Plot the density
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(models), figsize=(2*len(models)+1.5, 6))
    ticks = np.array([1.5, 11.5,21.5,31.5, 41.5, 45.5])
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax     ', 'other']

    for i, model in enumerate(models):
        ax = axs[i]
        im = model.get_counts(ignore_merged=True) / len(model.series)
        im[3:5] += im[1:3]
        implot = ax.imshow(im[3:49, :15], vmin=0, vmax=0.9)
        ax.set_xlabel('t')
        if i == 0:
            ax.set_ylabel('item')
        ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
        ax.set_ylim(-0.5, 45.5)

    fig.colorbar(implot, location='right', ax=axs, shrink=0.75)

    fig.savefig(plots_folder / name, bbox_inches='tight')
    plt.close(fig)

def plot_warping_single_series(model, name, nr):
    """
    Series 53 = [DZ_00001_0000414_itemsets.txt', '10500']
    """
    ticks = np.array([1.5, 11.5, 21.5, 31.5, 41.5, 45.5]) - 2
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax    ', 'other']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    # before warping
    ax = axs[0]
    im = model.series[nr].T
    ax.imshow(1-im[5:49], cmap='gray')
    ax = add_grid(ax)
    ax.set_xlabel('t')
    ax.set_ylabel('item')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_xlim(-0.5, 24.5)
    ax.set_title('before warping')

    # after warping
    ax = axs[1]
    im = np.sign(model.warped_series[nr].T)
    ax.imshow(1-im[5:49], cmap='gray')
    ax = add_grid(ax)
    ax.set_xlabel('t')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_xlim(-0.5, 24.5)
    ax.set_title('after warping')

    fig.savefig(plots_folder / name, bbox_inches='tight')
    plt.close(fig)
    return fig,ax

def plot_examples_of_sequences(apnea_model, hypopnea_model, name, nr_apnea, nr_hypopnea):
    """
    Plot an example of an apnea and hypopnea sequence
    """
    ticks = np.array([ 1.5, 11.5, 21.5, 31.5, 41.5, 45.5])
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax    ', 'other']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    # before warping
    ax = axs[0]
    im = apnea_model.series[nr_apnea].T
    im[3:5] += im[1:3]
    ax.imshow(1 - im[3:49], cmap='gray')
    ax.grid(axis='y', linewidth=2, color='k')
    for x in range(27):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5)
    for y in range(48):
        ax.axhline(y - 1.5, color='gray', linewidth=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('item')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 45.5)
    ax.set_xlim(-0.5, 20.5)

    # after warping
    ax = axs[1]
    im = hypopnea_model.series[nr_hypopnea].T
    ax.imshow(1 - im[3:49], cmap='gray')
    ax.grid(axis='y', linewidth=2, color='k')
    for x in range(27):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5)
    for y in range(48):
        ax.axhline(y - 1.5, color='gray', linewidth=0.5)
    ax.set_xlabel('t')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 45.5)
    ax.set_xlim(-0.5, 20.5)

    fig.savefig(plots_folder / name, bbox_inches='tight')
    plt.close(fig)
    return fig, ax

def plot_densities_toy_example(filename):
    xnew = np.linspace(0, 5, 300)

    fn = Path(__file__).parent.parent / "tests" / "rsrc" / "example9.txt"
    es = EventSeries.from_file(fn, window=LinearScalingWindow(3), constraints=[])

    fn2 = Path(__file__).parent.parent / "tests" / "rsrc" / "example9_aligned.txt"
    es2 = EventSeries.from_file(fn2, window=LinearScalingWindow(3), constraints=[])

    symbol = [0, 1]
    es.compute_windowed_counts()
    es.compute_rescaled_counts()
    es.compute_warping_directions()

    es2.compute_windowed_counts()
    es2.compute_rescaled_counts()
    es2.compute_warping_directions()

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row',
                            figsize=(10, 4))
    cnts = es.get_counts(ignore_merged=True)
    cnts2 = es2.get_counts(ignore_merged=True)
    colors = [c["color"] for c in matplotlib.rcParams["axes.prop_cycle"]]
    for curidx, cursymbol in enumerate(symbol):
        curcnts = cnts[cursymbol]
        curcnts2 = cnts2[cursymbol]

        # initial
        ax = axs[0, curidx]
        if curidx ==0:
            ax.set_ylabel('Before alignment')
        ax.set_title(f"Symbol {es.int2symbol.get(cursymbol, cursymbol)}")
        ax.bar([1,2,3,4,5,6], curcnts/sum(curcnts), color=colors[0], label="Normalized counts")
        smooth = make_interp_spline(range(6), np.convolve(curcnts / sum(curcnts), [0.5, 1, 0.5], 'same') / 2)(xnew)
        ax.plot(xnew+1, smooth, '-', color=colors[3], label="Density $d_{i,S}$")
        ax.set_ylim(0,1)

        # Aligned
        ax = axs[1, curidx]
        if curidx ==0:
            ax.set_ylabel('After alignment')
        ax.set_xlabel('Timestamp t')
        ax.bar([1,2,3,4,5,6], curcnts2/sum(curcnts2), color=colors[0], label="Normalized counts")
        smooth = make_interp_spline(range(6), np.convolve(curcnts2 / sum(curcnts2), [0.5, 1, 0.5], 'same') / 2)(xnew)
        ax.plot(xnew+1, smooth, '-', color=colors[3], label="Density $d_{i,S}$")
        ax.set_ylim(0, 1)

    if filename is not None:
        fig.savefig(plots_folder / filename, bbox_inches='tight')
        plt.close(fig)

def plot_costs(models):
    fig, ax = plt.subplots(figsize=(4,4))
    fig2, ax2 = plt.subplots(figsize=(4,4))
    for model in models:
        costs = np.array(model.costs)
        ax.plot(range(1,len(costs)+1), -costs[:,0])  # rewards
        ax2.plot(range(1,len(costs)+1), costs[:,2])  # entropy
    ax.set_xlabel('iteration')
    ax2.set_xlabel('iteration')
    ax.set_ylabel('reward')
    ax2.set_ylabel('entropy')
    ax.set_xticks([0,5,10,15,20])
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_yticks([120, 125, 130, 135])

    fig.savefig(plots_folder / 'plot_rewards', bbox_inches='tight')
    plt.close(fig)
    fig2.savefig(plots_folder / 'plot_entropy', bbox_inches='tight')
    plt.close(fig2)

def plot_examples_kernels():
    fig, ax = plt.subplots(figsize=(4, 4))

    x = np.arange(-5,6)
    y = (x*0).astype('float')
    y[3:8] += 0.5/5
    y[5] += 0.5
    z = 0.2*(np.cos(np.pi*x/10))**2
    ax.fill(x,y, alpha=0.5)
    ax.fill(x,z, alpha=0.5)

    fig.savefig(plots_folder / 'examples_kernels', bbox_inches='tight')
    plt.close(fig)