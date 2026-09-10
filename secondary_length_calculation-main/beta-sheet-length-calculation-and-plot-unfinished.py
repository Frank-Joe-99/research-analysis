# python version >=3.7

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable




def parse_args():
    description = 'To calculate the beta-sheet or beta-strucutre length, get the dictionary of lenth and beta-sheet/structure number and plot it. You should add those parameter:'
    parser = argparse.ArgumentParser(description=description)

    f = 'The .xpm file name of raw probability data.'
    parser.add_argument('-f', help=f)

    s = 'Calculate the beta-sheet or beta-strucutre length, default is beta-sheet length.'
    parser.add_argument('-s', help=s, default=False)

    
    args = parser.parse_args()
    return args





def deal_xpm_str(raw_str, beta_structure = False):
    '''
    ~   /* "Coil" */,
    E   /* "B-Sheet" */,
    B   /* "B-Bridge" */,
    S   /* "Bend" */,
    T   /* "Turn" */,
    H   /* "A-Helix" */,
    I   /* "5-Helix" */,
    G   /* "3-Helix" */,
    '''
    if beta_structure:
        transtab = raw_str.maketrans('BSTHIG','E~~~~~') # trans ! beta-structure to coil
    else:
        transtab = raw_str.maketrans('BSTHIG','~~~~~~') # trans ! beta-sheet to coil
    trans_str = raw_str.translate(transtab)
    trans_str = trans_str.translate({ord(letter): None for letter in '\"\,\n'})
    return trans_str


def read_xpm(file):
    str_data = []
    with open(file, 'r') as f:
        lines = f.readlines()[::-1]

        for line in lines:
            
            if line[0] == '\"':
                trans_line = deal_xpm_str(line)
                str_data.append(trans_line)
            else:
                break
    res_num = len(str_data)
    frame_num = len(str_data[1])
    return str_data, res_num, frame_num

def trans_str(str_data, res_num, frame_num):
    len_data = []
    for time in range(frame_num):
        split_list = ''.join((str_data[res][time] for res in range(res_num))).split('~')
        len_list = [len(i) for i in split_list if len(i)!=0]
        len_data += len_list
    return len_data



if __name__ == '__main__':
    args = parse_args()
    xpm_file = args.f
    path = args.f[:-4]
    beta_structure = args.s

    str_data, res_num, frame_num = read_xpm(xpm_file, beta_structure)
    len_data = trans_str(str_data, res_num, frame_num)
    data_dic = Counter(len_data)
    key_value_array = np.array([list(data_dic.keys()), list(data_dic.values())])

    print(key_value_array)
