import time
import os
import numpy as np

#Split the data in patches and subpatches
#Submitted file to dask should not be larger than 2GB,
#drastically slowes down speed

def split2boxes(length, width, n_boxes, direction='y'):
    sub_boxes = []
    if direction is 'x':
        sub_box_axes = width
    elif direction is 'y':
        sub_box_axes = length
        
    #split the data into boxes
    step = int(np.ceil(sub_box_axes / n_boxes))
    
    sub_boxes = []
    
    for i in range(n_boxes):
        c0 = 0 + step * i
        c1 = 0 + step * (i + 1)
        c1 = min(c1, sub_box_axes)
        if direction is 'x':
            sub_boxes.append(np.s_[0:length, c0:c1])
        elif direction is 'y':
            sub_boxes.append(np.s_[c0:c1, 0:width])
    return sub_boxes

def slice_func_data(fdata):
    keys = list(fdata.keys())
    func_data_new = {}
    for k in keys[:-1]:
        func_data_new[k] = fdata[k][fdata['box']]
    func_data_new['box'] = fdata['box']
    return func_data_new


class local_cluster:
    # local Dask cluster
    def __init__(self, num_workers, **kwargs):
        self.num_workers = num_workers
        self.cluster_kwargs = kwargs
        ## format input arguments
        self.cluster_kwargs['config_name'] = None
        
        self.cluster = None
        self.client = None
    def open(self):
        """Initiate the cluster"""
        print('initiate Dask cluster')
        from dask.distributed import LocalCluster, Client
        self.cluster = LocalCluster()

        print('scale Dask cluster to {} workers'.format(self.num_workers))
        self.cluster.scale(self.num_workers)

        print('initiate Dask client')
        self.client = Client(self.cluster)
        self.client.get_versions(check=True)
        print(self.client.dashboard_link)
        
        
    def run(self, func, func_data, results):
        """Wrapper function encapsulating submit_workers and compile_workers."""
        #from dask.distributed import Client
        
        dim, length, width = func_data['data'].shape
        print(dim, length, width)
        sub_boxes = split2boxes(length, width, self.num_workers, 'y')
        #Add 3 dim
        sub_boxes = [tuple(np.append(np.s_[:], i)) for i in sub_boxes] 
        
        # start a bunch of workers from the cluster
        #print('scale Dask cluster to {} workers'.format(self.num_workers))
        #self.cluster.scale(self.num_workers)

        #print('initiate Dask client')
        #self.client = Client(self.cluster)
        #self.client.get_versions(check=True)
        #print(self.client.dashboard_link)
        
        # submit job for each worker
        futures, submission_time = self.submit_job(func, func_data, sub_boxes)

        # assemble results from all workers
        results = self.collect_result(futures, results, submission_time)

        return results
    
    def submit_job(self, func, func_data, sub_boxes):
        submission_time = time.time()
        futures = []

        for i, sub_box in enumerate(sub_boxes):
            print(f'submit a job to the worker for sub box {i}: {sub_box}')
            func_data['box'] = sub_box

            func_data_new = slice_func_data(func_data)
            print('Spliting the func data')
            print(func_data['data'].shape, '>', func_data_new['data'].shape)

            future = self.client.submit(func, **func_data_new, retries=3)
            #future = self.client.submit(func, **func_data, retries=3)
            futures.append(future)

        return futures, submission_time
    
    def collect_result(self, futures, results, submission_time):
        from dask.distributed import as_completed
        
        num_future = 0
        for future, sub_results in as_completed(futures, with_results=True):
            # message
            
            num_future += 1
            sub_t = time.time() - submission_time
            print("FUTURE #{} complete. Time used: {:.0f} seconds".format(num_future, sub_t))
            
            # catch result - sub_box
            sub_box = sub_results[-1] #[1:]
            
            for i, sub_result in enumerate(sub_results[:-1]):
                if sub_result is not None:
                    num_dim = sub_result.ndim
                    if num_dim ==2:
                        sub_box = sub_box[1:] #reject the first dimension ndim

                    if num_dim is results[i].ndim:
                        results[i][sub_box] = sub_result
                    
                    else:
                        msg = f'worker result dimension: {num_dim} '
                        msg += f'does not match given results ndim {results[0].ndim}'
                        raise Exception(msg)

                    
        return results
    
    def close(self):
        """Close connections to dask client"""
    
        self.cluster.close()
        print('close dask cluster')
    
        self.client.close()
        print('close dask client')