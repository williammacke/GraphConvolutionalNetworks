# plot_max_memory.py

import json
import matplotlib.pyplot as plt
import seaborn as sns


def main():
  with open('max_memory.json', 'r') as f:
    used_memory = json.load(f)
  tkipf = used_memory['tkipf']
  dgl   = used_memory['dgl']
  ours  = used_memory['ours']
  x = ['tkipf', 'dgl', 'ours']
  y = [tkipf, dgl, ours]
  ax = sns.barplot(x=x, y=y)
  ax.set(ylabel='Max GPU memory used (MiB)')
  plt.savefig('max_memory.png')
  plt.show()


if __name__ == '__main__':
    main()
