import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable




def array_deall(array_mat, t = 310, s_min = False):
    r = 0.0019863  # kcal/mol
    min_array = np.sort(np.ndarray.flatten(array_mat))

    if s_min == False:
        for i in min_array:
            s_min = -1
            if i != 0 and i > 0:
                s_min = i
                break
        array_deal = array_mat + s_min
        print(s_min)
    else:
        array_deal = array_mat + s_min
        print(s_min)
        
    array_fin = -1 * r * t * (np.log(array_deal) - np.log(s_min))
    return array_fin, round(array_fin.min())

def parse_args():
    description = 'To calculate the energy of pi-pi stacking and plot it. You should add those parameter:'
    parser = argparse.ArgumentParser(description=description)

    f = 'The file name of raw probability data.'
    parser.add_argument('-f', help=f)

    t = 'The tempreture to calculate the free energy, the default tempreture is 310 K.'
    parser.add_argument('-t', help=t, default=310)

    b = 'The color group of colorbar, default (1) is cyan-green-yello-red. \n Support color groups: \n (1) cyan-green-yello-red \n (2) white-cyan-yello-red \n (3) white-red-orange-purple \n (4) blue-green-yellow-red-purple '
    parser.add_argument('-b', help=b, default='1')

    c = 'The minimum value of colorbar, the default value is using the round function to approximate the minimum value of the energy matrix. If you do not like the colorbar\'s range, you can set this parameter by yourself.'
    parser.add_argument('-c', help=c, default=False)

    d = 'The minimum value of matrix. To keep different FELs comparable. It should be the largest value amonge those minimum values.'
    parser.add_argument('-d', help=d, default=False)

    e = 'Whether print raw energy file or not. Default is do not print.'
    parser.add_argument('-e', help=e, action='store_true')

    i = 'The interpolation method used. Default value is: bilinear. Supported values are: none, antialiased, nearest, bilinear, bicubic, spline16, spline36, hanning, hamming, hermite, kaiser, quadric, catrom, gaussian, bessel, mitchell, sinc, lanczos, blackman. For more information, please check https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html'
    parser.add_argument('-i', help=i, default='bilinear')

    n = 'The number of colorbar quantization levels, the default value is 40.'
    parser.add_argument('-n', help=n, default=40)

    p = 'Photo format, default format is .svg. Supported formats are: png, jpg, eps, svg.'
    parser.add_argument('-p', help=p, default='svg')

    z = 'font chosen. default is Times New Roman. you can choose: 1: Times New Roman. 2: Arial. '
    parser.add_argument('-z', help=z, default=1)

    
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':

    args = parse_args()
    path = args.f[:-4]
    temp = float(args.t)
    N = int(args.n)
    interpolation = args.i
    photo_format = str(args.p)
    colorgroup = int(args.b)-1
    font = int(args.z)-1

    font_list = ['Times New Roman', 'Arial']
    mpl.rcParams['font.family'] = font_list[font]
    mpl.rcParams['font.sans-serif'] = font_list[font]
    mpl.rcParams['mathtext.fontset'] = 'custom'

    if args.d != False:
        s_min = float(args.d)
    else:
        s_min = args.d


    mycolor = [['#00FFFF','#00FF00','#FFFF00','#FF0000'],
               ['#FFFFFF','#00FFFF','#FFFF00','#FF0000'],
               ['#FFFFFF','#F2220F','#F28907','#F2B807','#5F49F2'],
               ['#419df1','#64e7eb', '#6dfa3d', '#00d800', '#019000', '#ffff00', '#e7c000', '#ff9000', '#ff0000', '#d60000', '#c00000', '#ff00f0', '#9600b4']]

    #mycolor = ['#00FFFF','#00FF00','#FFFF00','#FF0000']
    
    mycmap = colors.LinearSegmentedColormap.from_list('my_list', mycolor[colorgroup][::-1], N)
    
    f = open(path+'.dat', 'r')
    data = []
    for line in f:
        l = list(map(float, line.split()))
        data.append(l)
    f.close()

    data_in = np.array(data)
    data_fin, value_min = array_deall(data_in.transpose(), temp, s_min)
    if args.c != False:
        value_min = float(args.c)

    if args.e:
        data_save = open(path+'-energy.dat', 'w')
        for s in data_fin:
            for k in s:
                data_save.writelines(str(k) + '\t')
            data_save.writelines('\n')
        data_save.close()
        print('energy data is saved as:', path+'.dat')

    y = [0, 30, 60, 90]
    y_90 = [0, 30, 60, 90]
    y_180 = [0, 60, 120, 180]

    if data_fin.shape[1] == 90.0 or data_fin.shape[1] == 900.0:
        plt.figure(figsize=(1.8,1.8))
    elif data_fin.shape[1] == 160.0 or data_fin.shape[1] == 1600.0:
        plt.figure(figsize=(2.3,1.8))
    elif data_fin.shape[1] == 250.0 or data_fin.shape[1] == 2500.0:
        plt.figure(figsize=(2.8,1.8))
    
    
    ax = plt.subplot(111)
    ax.tick_params(direction='in')

    if data_fin.shape[1] == 90.0 :
        im = plt.imshow(data_fin,origin='lower',aspect='auto',extent=(0,data_fin.shape[1]/100,0,90),
                   interpolation=interpolation, vmin=value_min, vmax=0, cmap=mycmap)

    elif data_fin.shape[1] == 120.0:
        im = plt.imshow(data_fin,origin='lower',aspect='auto',extent=(0,data_fin.shape[1]/100,0,120),
                   interpolation=interpolation, vmin=value_min, vmax=0, cmap=mycmap)

    elif data_fin.shape[1] == 900.0:
        im = plt.imshow(data_fin,origin='lower',aspect='auto',extent=(0,data_fin.shape[1]/1000,0,data_fin.shape[0]/10),
                   interpolation=interpolation, vmin=value_min, vmax=0, cmap=mycmap)

    elif data_fin.shape[1] == 1200.0:
        im = plt.imshow(data_fin,origin='lower',aspect='auto',extent=(0,data_fin.shape[1]/1000,0,data_fin.shape[0]/10),
                   interpolation=interpolation, vmin=value_min, vmax=0, cmap=mycmap)

    elif data_fin.shape[1] == 160.0 or data_fin.shape[1] == 250.0:
        im = plt.imshow(data_fin,origin='lower',aspect='auto',extent=(0,data_fin.shape[1]/100,0,180),
                   interpolation=interpolation, vmin=value_min, vmax=0, cmap=mycmap)

    elif data_fin.shape[1] == 1600.0 or data_fin.shape[1] == 2500.0:
        im = plt.imshow(data_fin,origin='lower',aspect='auto',extent=(0,data_fin.shape[1]/1000,0,data_fin.shape[0]/10),
                   interpolation=interpolation, vmin=value_min, vmax=0, cmap=mycmap)


    plt.xlabel('Distance (nm)', fontsize=12)
    plt.ylabel('Angle ($\degree$)', fontsize=12)
    #plt.title('kcal/mol', x=1.1, y=1, fontsize=11)

    if data_fin.shape[1] == 90 or data_fin.shape[1] == 900:
        plt.xticks([0, 0.2, 0.4, 0.6], fontsize=11)
        plt.xlim(0, 0.7)
        plt.yticks(y_90, fontsize=11)

    elif data_fin.shape[1] == 120 or data_fin.shape[1] == 1200:
        plt.xticks([0, 0.4, 0.8, 1.2], fontsize=11)
        plt.xlim(0, 1.2)
        plt.yticks(y_90, fontsize=11)

    elif data_fin.shape[1] == 160.0 or data_fin.shape[1] == 1600.0:
        plt.xticks([0, 0.4, 0.8, 1.2, 1.6], fontsize=11)
        plt.xlim(0, 1.6)
        plt.yticks(y_180, fontsize=11)
        
    elif data_fin.shape[1] == 250.0 or data_fin.shape[1] == 2500.0:
        plt.xticks([0, 0.5, 1.0, 1.5, 2.0,2.5], fontsize=11)
        plt.xlim(0, 2.5)
        plt.yticks(y_180, fontsize=11)
    else:
        plt.xticks([0, 0.2, 0.4, 0.6], fontsize=11)
        plt.xlim(0, 0.7)
        plt.yticks(y_90, fontsize=11)


    #plt.yticks(y, fontsize=11)
    plt.rcParams['font.size'] = 11

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="5%")
    plt.colorbar(im, cax=cax).ax.set_title('kcal/mol', fontsize=11)

    
    plt.savefig(path+'-stacking.'+photo_format, bbox_inches='tight')
    #plt.savefig(path+'-stacking.'+photo_format)
    plt.close()
    print('plot is saved as:', path + '-stacking.' + photo_format)
