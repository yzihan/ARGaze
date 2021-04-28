import torch
from torch import nn
import numpy as np
#import cv2
from PIL import Image

# compatibility purpose...
def read_image(path):
#    return cv2.imread(path)
    return np.array(Image.open(path))
    
def __parse_list(val):
    if len(val) == 1:
        return { int(val[0]) }
    elif len(val) == 2:
        a, b = int(val[0]), int(val[1])
        return set([ x for x in range(a, b + 1) ])
    else:
        print('Invalid list format: {}'.format(val))
        exit(1)

def parse_list(val):
    aa = set()
    for x in map(lambda val: __parse_list(val.split('-')), val.split(',')):
        aa = aa.union(x)
    return sorted(list(aa))

def print_modelsize(model, type_size=4):
#    for p in model.parameters():
#        print(p.size())
    param_size = sum([np.prod(list(p.size())) for p in model.parameters()]) * type_size
    unit = 'B'
    if param_size > 1024:
        param_size /= 1024
        unit = 'KiB'
        if param_size > 1024:
            param_size /= 1024
            unit = 'MiB'
            if param_size > 1024:
                param_size /= 1024
                unit = 'GiB'
    print('Model parameter size: {:6f}{}'.format(param_size, unit))
