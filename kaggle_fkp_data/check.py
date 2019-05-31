import h5py
import cv2
import numpy as np
#from vis_util import draw_point

def show(img, *point):
    image = np.swapaxes(img, 0, 1)
    image = np.swapaxes(image, 1, 2)
    #image = draw_point(image, (x1, y1, x2, y2, x3, y3, x4, y4))
    cv2.imshow('a', image)
    cv2.waitKey()
    return 

def check(*args):
    if len(args) == 8:
        return _check(*args)
    return False

def _check(x1, y1, x2, y2, x3, y3, x4, y4):
    if x1 < x2 and x4 < x3 and y1 < y4 and y2 < y3 and x1 < x3 and x4 < x2:
        return True
    return False

if __name__ == '__main__':
    f = h5py.File('./train_data.h5', 'r')
    for i, (image, (x1, y1, x2, y2, x3, y3, x4, y4)) in enumerate(zip(f['data'], f['label'])):
        print(i)
        show(image, (x1, y1, x2, y2, x3, y3, x4, y4))
        if not check(x1, y1, x2, y2, x3, y3, x4, y4):
            print(i)
            show(image, (x1, y1, x2, y2, x3, y3, x4, y4))
            print(x1, y1, x2, y2, x3, y3, x4, y4)

