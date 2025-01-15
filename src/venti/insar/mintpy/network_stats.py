#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime 
import networkx as nx
from mintpy.objects import ifgramStack

def create_parser():
    parser = argparse.ArgumentParser(description=
        'Plot count ifgs per acquisition date')
    parser.add_argument('-i', '--ifgs_stack', type=str,
        help='Input Mintpy ifgramStack.h5', dest='ifgram_file')
    return parser

def colored_str(msg):
    return '\x1b[6;30;42m' + msg + '\x1b[0m' 

def get_stats(ifgram_file):
    try:
        stack_obj = ifgramStack(Path(ifgram_file).resolve())
        stack_obj.open()
    except:
        raise ValueError(f'Cannot open {ifgram_file}')

    # Get dates
    date12List = stack_obj.get_date12_list(dropIfgram=True)
    dates12array = np.vstack([d.split('_') for d in date12List])
    
    edges = [(d1, d2) for d1, d2 in zip(dates12array[:, 0],
                                        dates12array[:, 1])]
    count_list = np.dstack(np.unique(dates12array, return_counts=True))

   # Print count per date
    print('  DATE       COUNT')
    print(20*'_')
    for count_ix in np.squeeze(count_list):
        msg = count_ix[0] + '   :   ' + count_ix[1] 
        if int(count_ix[1]) == 1:
            print(colored_str(msg))
        else:
            print(msg) 
    print(20*'_')
    # Check if network is connected
    G = nx.Graph()
    G.add_edges_from(edges)
    if nx.is_connected(G):
        print('Network is connected!')
    else:
        msg = 'Network is disconnected'
        print(colored_str(msg)) 
        for ix, cluster in enumerate(list(nx.connected_components(G))):
            print(f'Cluster {ix}: {len(cluster)}')
            print(cluster, '\n')

def main(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    ifg_input = Path(inps.ifgram_file).resolve()
    
    get_stats(ifg_input)
    

if __name__ == '__main__':
    main(sys.argv[1:])
