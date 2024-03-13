import bcolz
import numpy as np
import os
import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame

def get_val_pair(path, name, use_memfile=True):
    if use_memfile:
        mem_file_dir = os.path.join(path, name, 'memfile')
        mem_file_name = os.path.join(mem_file_dir, 'mem_file.dat')
        if os.path.isdir(mem_file_dir):
            print('laoding validation data memfile')
            np_array = read_memmap(mem_file_name)
        else:
            os.makedirs(mem_file_dir)
            carray = bcolz.carray(rootdir = os.path.join(path, name), mode='r')
            np_array = np.array(carray)
            mem_array = make_memmap(mem_file_name, np_array)
            del np_array, mem_array 
            np_array = read_memmap(mem_file_name)
    else:
        np_array = bcolz.carray(rootdir = os.path.join(path, name), mode='r')

    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
    return np_array, issame


def make_memmap(mem_file_name, np_to_copy):
    memmap_configs = dict() 
    memmap_configs['shape'] = shape = tuple(np_to_copy.shape) 
    memmap_configs['dtype'] = dtype = str(np_to_copy.dtype)   
    json.dump(memmap_configs, open(mem_file_name+'.conf', 'w')) 
    # w+ mode: Create or overwrite existing file for reading and writing
    mm = np.memmap(mem_file_name, mode='w+', shape=shape, dtype=dtype)
    mm[:] = np_to_copy[:]
    mm.flush() # memmap data flush
    return mm

def read_memmap(mem_file_name):
    # r+ mode: Open existing file for reading and writing
    with open(mem_file_name+'.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+', \
                         shape=tuple(memmap_configs['shape']), \
                         dtype=memmap_configs['dtype'])