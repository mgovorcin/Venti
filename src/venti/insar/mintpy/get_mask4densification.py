#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from pathlib import Path
from mintpy.utils import readfile, writefile

def create_parser():
    parser = argparse.ArgumentParser(description=
        'Get mask from timeseries for densification')
    parser.add_argument('-i', type=str,
        help='Input Mintpy timeseries*.h5', dest='ts_file')
    parser.add_argument('-o', '--output', default='./maskTS.h5',
                        help='Output dir', type=str, dest='output')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Display print messages', dest='verbose')
    return parser

def main(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # Generate mask from ts
    data, atr = readfile.read(str(inps.ts_file),
                            datasetName='timeseries')

    # Get shape
    num_date, length, width = data.shape

    # Reshape
    data2 = data.reshape((num_date, -1))
    count = np.count_nonzero(data2, axis=0)
    del data2

    # Get mask
    mask = np.zeros(count.shape, dtype=np.bool_)
    # remove first date as it is ref filled with zeros
    mask[np.where(count == (num_date - 1))] = True
    mask = mask.reshape((length, width))

    # Add ref
    ref_yx = [np.int16(atr['REF_Y']),
              np.int16(atr['REF_X'])]

    mask[ref_yx] = True

    # Write
    atr['FILE_TYPE'] = 'mask'
    writefile.write(~mask, out_file=inps.output, metadata=atr)

if __name__ == '__main__':
    main(sys.argv[1:])