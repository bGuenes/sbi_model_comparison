import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#import jax
#import jax.numpy as jnp
import numpy as np
import multiprocessing as mp


def pairplot_sns(i):
    print(i)

    s = pd.DataFrame(samples[:,i,:], columns=labels_in+labels_out)

    plot = sns.pairplot(s, kind="kde")

    for j in range(5):
        plot.axes[j, j].set_xlim(-5, 5)
        plot.axes[j, j].set_ylim(-5, 5)
    plot.axes[5, 5].set_xlim(-5, 15)
    for j in range(6,14):
        plot.axes[j,j].set_xlim(-2,2)
        plot.axes[j,j].set_ylim(-2,2)

    plot.savefig(f"sns4/pairplot_{i:03d}.png")
    plt.clf()
    plt.close()



labels_out = ['C', 'Fe', 'He', 'Mg', 'N', 'Ne', 'O', 'Si']
labels_in = ['high_mass_slope', 'log10_N_0', 'log10_starformation_efficiency', 'log10_sfr_scale', 'outflow_feedback_fraction', 'time']

samples = np.load("samples.npy")

"""s = np.zeros((len(samples[0]),), dtype=object)
for i in range (len(samples[0])):
    s[i] = pd.DataFrame(samples[:,i,:], columns=labels_in+labels_out)"""

with mp.Pool() as pool:
    pool.map(pairplot_sns, np.arange(len(samples[0])))