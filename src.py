# import tensorflow.compat.v2 as tf
import numpy as np
import glob
from PIL import Image

'''
    load_dataset
    ###############
    Load dataset at base path specified

    Return values
    ===============
    data: nx80x80 numpy array of type float, samples are averaged over RGB channels
    label: corresponding int label of the samples
'''
def load_dataset(path):
    files = [f for f in glob.glob(path + '*.jpg')]
    labels = np.zeros((len(files), 1), dtype = np.int32)
    data = np.zeros((len(files), 80,80), dtype = np.float)
    idx = 0
    for file in files:
        name = file.split('/')[-1].split('.jpg')[0]
        labels[idx] = int(name.split('_')[0])
        data[idx,:] = np.mean(np.asarray(Image.open(file), dtype = np.float), axis = 2)
    return data, labels

data, labels = load_dataset('data/training/')
