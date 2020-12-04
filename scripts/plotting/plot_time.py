# plot_time.py

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    with open('times.json', 'r') as f:
        times = json.load(f)
    tkipf_mean = np.mean(times['tkipf'])
    dgl_mean   = np.mean(times['dgl'])
    ours_mean  = np.mean(times['ours'])
    x = ['tkipf', 'dgl', 'ours']
    y = [tkipf_mean, dgl_mean, ours_mean]
    ax = sns.barplot(x=x, y=y)
    ax.set(ylabel='Average time per epoch (ms)')
    plt.savefig('time_per_epoch.png')
    plt.show()


if __name__ == '__main__':
    main()
