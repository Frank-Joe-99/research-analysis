import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5

 
filename = './cluster_distribution.npz'
data = np.load(filename)['cluster_distributions']
print(data.shape)
print(data.max())
# print(data['cluster_distributions'].shape)
# (chain_num, frame)
# (217, 1000)

'''
bin_size = 1
n = data.shape[1] // bin_size

data_plot = data[:, :n * bin_size].reshape(
    data.shape[0], n, bin_size
).max(axis=2)
'''

x  = np.array([0.0, 0.02, 0.2, 2,   20])
x_ = np.array([1,   40,  400,  4000, 40000])

y = np.array([1, 50, 100, 150, 200])


mycolor = ["#880000",'#FF0000','#FFFFFF']
mycmap = colors.LinearSegmentedColormap.from_list('my_list', mycolor[::-1], 255)


plt.figure(figsize=(3, 2))
ax = plt.subplot(111)
ax.tick_params(direction='in', pad=3, length=2, which='both', bottom=True, top=True, left=True, right=True)
# im = plt.imshow(data_plot, origin='lower', vmin=0, cmap=mycmap, vmax=1, aspect='auto', interpolation='nearest')

x_edges = np.arange(1, data.shape[1] + 2)
y_edges = np.arange(data.shape[0] + 1)
im = ax.pcolormesh(x_edges, y_edges, data, shading="auto", vmin=0, cmap=mycmap, vmax=1, rasterized=True)
ax.set_xscale("log")


plt.xticks(x_, x, fontsize=8)
plt.yticks(y, y, fontsize=8)
plt.title('Cluster size distribution', fontsize = 9)

plt.xlabel(r'Time ($\mu$s)', fontsize=9)
plt.ylabel('Cluster size', fontsize=9)

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", 0.05, pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=8, pad=3, length=2)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels([0, 0.5, 1])

plt.savefig('./cluster-distribution.svg', bbox_inches='tight', dpi=300)
