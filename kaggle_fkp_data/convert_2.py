import os
import numpy as np
from sklearn.utils import shuffle
import h5py
import glob
import cv2
import sys
import random
from cc_utils.file import load_lines
from check import check
def load(img_names, kps):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    X = []
    Y = []
    for img_name, kp in zip(img_names, kps): 
        label = np.array(map(float, kp))
        img = cv2.imread(img_name)
        h, w, c = img.shape
        
        if not check(*label):
            continue

        x_max = int(max(label[::2]))
        y_max = int(max(label[1::2]))
        x_min = int(min(label[::2]))
        y_min = int(min(label[1::2]))

        if y_max - y_min < 20:
            print('Skip: height', y_max - y_min)
            continue

        new_x_min = max(int(x_min) - random.randint(1, 40), 0)
        new_y_min = max(int(y_min) - random.randint(1, 40), 0)
        new_x_max = min(int(x_max) + random.randint(1, 40), w)
        new_y_max = min(int(y_max) + random.randint(1, 40), h)

        label[::2] -= new_x_min
        label[1::2] -= new_y_min
        label[::2] /= (new_x_max - new_x_min)
        label[::2]  *= 128
        label[1::2] /= (new_y_max - new_y_min)
        label[1::2] *= 64

        img = img[new_y_min:new_y_max, new_x_min:new_x_max]
        try:
            img = cv2.resize(img, (128, 64))
        except Exception:
            print('ERR, resize:', img.shape)
            continue

        X.append(img)
        Y.append(label)
    X = np.stack(X, 0)
    Y = np.stack(Y, 0)
    print(X.shape, Y.shape)

    X, Y = shuffle(X, Y)  # shuffle train data
    Y = Y.astype(np.float32)

    return X, Y

def writeHdf5(t,data,label=None, suffix=''):
    with h5py.File(os.getcwd()+ '/'+t + '_data%s.h5'%suffix, 'w') as f:
        f['data'] = data
        if label is not None:
            f['label'] = label
            with open(os.getcwd()+ '/'+t + '_data_list%s.txt'%suffix, 'w') as f:
                f.write(os.getcwd()+ '/' +t + '_data%s.h5\n'%suffix)

import json
img_names = glob.glob('/ssd/zq/parkinglot_pipeline/carplate/crops_all/data/WANDA/20180921/crops/*.jpg')
json_file = json.load(open('/mnt/soulfs2/fzhou/data/wanda/label/20181119_carplate_wanda_0921.json'))
img = []
pts = []
for i in img_names:
    if os.path.basename(i) in json_file:
        pt = json_file[os.path.basename(i)]
        if len(pt) > 0:
            img.append(i)
            pts.append(pt[0]['coordinates'][:8])
X, y = load(img, pts)
X = np.swapaxes(X, 2, 3)
X = np.swapaxes(X, 1, 2)
print(X.shape)
#sep = 679000000
#writeHdf5('train',X[0:sep],y[0:sep], sys.argv[1])
#writeHdf5('val',X[sep:],y[sep:], sys.argv[1])
writeHdf5('train',X,y, sys.argv[1])
