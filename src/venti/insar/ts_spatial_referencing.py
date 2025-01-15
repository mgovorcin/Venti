#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
from mintpy.objects import timeseries
from mintpy.objects import cluster
from mintpy.utils import readfile, writefile, ptime


def create_parser():
    parser = argparse.ArgumentParser(
        description='Remove noise related with reference point by using temporal average')
    parser.add_argument('timeseries_file',
                        help='Time series file')
    parser.add_argument('-o', '--output', default=None,
                    help='Output dir', type=str, dest='output')
    parser.add_argument('--step', '--step-date', dest='steps',
                             type=str, nargs='+', default=[],
                             help='step function(s) at YYYYMMDD (default: %(default)s). E.g.:\n'
                             '--step 20061014          # coseismic step  at 2006-10-14T00:00')
    return parser

def _reference_with_avg(ts_data, ts_adj):
    # Get temporal average
    ts_avg = np.nanmean(np.ma.masked_equal(ts_adj, 0), axis=0)

    # Subtract temporal average from timeseries
    corr_ts = ts_data - ts_avg

    # Replace first data with all zeros
    corr_ts[0] = ts_data[0]
    return corr_ts

def _adjust4steps(ts_data, date_list, steps=[]):
    ts = ts_data.copy()
    if len(steps) > 0:
        dates = np.array(ptime.date_list2vector(date_list)[1])

        # Find steps indexes
        steps = ptime.date_list2vector(steps)[1]
        steps_ix = [np.max(np.where(dates < step)) for step in steps]

        # add the end
        steps_ix.append(len(date_list))

        # find offsets
        offsets = [ts_data[step + 1, :, :] - ts_data[step, :, :]
                   for step in steps_ix[:-1]]
        for i, step in enumerate(steps_ix[:-1]):
            iy = np.s_[step:steps_ix[i+1]]
            ts[iy] = ts[iy] - offsets[i]

    return ts


def main(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # Read timeseries
    # NOTE: takes a lot of memory run it in patches
    print(f' Getting info from: {inps.timeseries_file}')
    attr = readfile.read_attribute(inps.timeseries_file)
    print(inps.steps)

    # Get dates and bperp
    refobj = timeseries(inps.timeseries_file)
    refobj.open(print_msg=False)
    dates_list = refobj.dateList
    dates = np.array(dates_list, dtype=np.string_)
    num_date = len(dates)
    pbase = refobj.pbase
    del refobj

    length = int(attr.get('LENGTH'))
    width = int(attr.get('WIDTH'))

    memoryAll = (num_date + 2) * length * width * 4
    num_box = int(np.ceil(memoryAll * 3 / (4 * 1024**3)))
    box_list, num_box = cluster.split_box2sub_boxes(
        box=(0, 0, width, length),
        num_split=num_box,
        dimension='y',
        print_msg=True,
    )

    # Write it down
    ds_dict = {
        "date"       : [np.dtype("S8"), (num_date,), dates],
        "bperp"      : [np.float32,     (num_date,), pbase],
        "timeseries" : [np.float32,     (num_date, length, width)],
    }
    if inps.output is None:
        output = Path(inps.timeseries_file).resolve()
        name = output.name.split('.')[0] + '_ref.h5'
        inps.output = output.parents[0] / name

    print(f'Write rereferenced timeseries to: {inps.output}')
    writefile.layout_hdf5(inps.output,
                      metadata=attr,
                      ds_name_dict=ds_dict)

    # Rereference
    for ix, box in enumerate(box_list):
        print(f' Run box {ix+1} of {len(box_list)}')
        ts_data = readfile.read(inps.timeseries_file, box=box)[0]
        ts_adj = _adjust4steps(ts_data, dates_list, inps.steps)
        ts_ref = _reference_with_avg(ts_data, ts_adj)
        block = [0, num_date, box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(inps.output,
                        data=ts_ref,
                        block=block,
                        datasetName='timeseries')

if __name__ == '__main__':
    main(sys.argv[1:])