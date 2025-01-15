#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime 
from mintpy.objects import ifgramStack
from mintpy.utils import readfile, writefile

def create_parser():
    parser = argparse.ArgumentParser(description=
        'Plot count ifgs per acquisition date')
    parser.add_argument('-i', '--ifgs_stack', type=str,
        help='Input Mintpy ifgramStack.h5', dest='ifgram_file')
    parser.add_argument('-o', '--output', default='./date_count.h5',
                        help='Output dir', type=str, dest='output')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Display print messages', dest='verbose')
    return parser


def get_h5_count(ifgram_file, output, verbose=False):
    try:
        stack_obj = ifgramStack(Path(ifgram_file).resolve())
        stack_obj.open()
    except:
        raise ValueError(f'Cannot open {ifgram_file}')

    # Get dates
    date_list = stack_obj.get_date_list(dropIfgram=True)
    date12List = stack_obj.get_date12_list(dropIfgram=True)
    dates12array = np.vstack([d.split('_') for d in date12List])
    
    # Get metadata
    meta = stack_obj.get_metadata()
    meta['FILE_TYPE'] = 'mask'
    meta['UNIT'] = '1' 
    meta['REF_DATE'] = date_list[0]
    num_date = len(date_list)
    length, width = stack_obj.get_size()[1:]


    # Instantiate time-series
    dates = np.array(date_list, dtype=np.string_)
    ds_name_dict = {
        "date"       : [dates.dtype, (num_date,), dates],
        "timeseries" : [np.float32,  (num_date, length, width), None],
    }
    writefile.layout_hdf5(str(output), ds_name_dict, metadata=meta)

    # Get count
    with tqdm(total=num_date) as pbar:
        for ik, date in enumerate(date_list):
            # Find pairs with selected date
            index = np.where(date == dates12array)[0]
            select_ix = [date12List[ix] for ix in index]

            if len(select_ix) < 2:
                print(f'Skip pair with 1 connection: {date}')
                continue
            # Read only pairs with selected date
            selection_array = readfile.read(ifgram_file, 
                                            datasetName=select_ix)[0]
            # Write to file
            block = [ik, ik+1, 0, length, 0, width]
            data = np.count_nonzero(selection_array, axis=0)
            writefile.write_hdf5_block(str(output), data=data, 
                                    datasetName='timeseries',
                                    block=block, print_msg=False)
            pbar.update(1)
            pbar.set_description("Processing %s" % date)


def main(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    ifg_input = Path(inps.ifgram_file).resolve()
    output = Path(inps.output).resolve()
    
    print('Getting count per SAR acquisition date.')
    get_h5_count(ifg_input, output)
    

if __name__ == '__main__':
    main(sys.argv[1:])