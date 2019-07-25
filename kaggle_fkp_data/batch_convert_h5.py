#coding=utf-8
import os
import numpy as np
from sklearn.utils import shuffle
from multiprocessing import Process
import h5py
import glob
import json
import cv2
import sys
import random
import argparse
from check import check
import pdb
exclude_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ非缺无'.decode('utf8')

def process(h5_name, data_tuple):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    img_names, kps = data_tuple
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
        label[::2]  *= 224
        label[1::2] /= (new_y_max - new_y_min)
        label[1::2] *= 224

        img = img[new_y_min:new_y_max, new_x_min:new_x_max]
        try:
            img = cv2.resize(img, (224, 224))
        except Exception:
            print('ERR, resize:', img.shape)
            continue

        X.append(img)
        Y.append(label)

    X = np.stack(X, 0)
    Y = np.stack(Y, 0)

    print(X.shape, Y.shape)
    X = np.swapaxes(X, 2, 3)
    X = np.swapaxes(X, 1, 2)

    #X, Y = shuffle(X, Y)  # shuffle train data
    Y = Y.astype(np.float32)

    with h5py.File(h5_name, 'w') as f:
        f['data'] = X
        if Y is not None:
            f['label'] = Y
        print '=== H5 data written to ', h5_name

def writeHdf5(t,data,label=None, suffix=''):
    with h5py.File(os.getcwd()+ '/'+t + '_data%s.h5'%suffix, 'w') as f:
        f['data'] = data
        if label is not None:
            f['label'] = label
            with open(os.getcwd()+ '/'+t + '_data_list%s.txt'%suffix, 'w') as f:
                f.write(os.getcwd()+ '/' +t + '_data%s.h5\n'%suffix)

def writeHdf5Batch(t,data,label=None, suffix='', batch_size=5000):
    rest_size = data.shape[0]
    batch_num = 1
    with open(os.getcwd()+ '/'+t + '_data_list%s.txt'%suffix, 'w') as ft:
        while rest_size > 0:
            with h5py.File(os.getcwd()+ '/'+t + '_data%s_part%02d.h5'% (suffix,batch_num), 'w') as f:
                last_idx = (batch_num-1)*batch_size
                cur_idx = min(rest_size+last_idx, batch_num*batch_size)
                f['data'] = data[last_idx:cur_idx]
                if label is not None:
                    f['label'] = label[last_idx:cur_idx]
            ft.write(os.getcwd()+ '/' +t + '_data%s_part%02d.h5\n' % (suffix,batch_num))
            rest_size -= batch_size
            batch_num += 1


def write_image_info_into_hdf5(file_name, data_tuple, phase):
    total_size = len(data_tuple)
    print '[+] total image for {0} is {1}'.format(file_name, len(data_tuple))
    single_size = 5000
    groups = total_size / single_size
    if total_size % single_size:
        groups += 1
    """
    def test(file_name, data):
        img_data = np.zeros((len(data_tuple), 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype = np.float32)
        label_seq = 73*np.ones((len(data_tuple), LABEL_SEQ_LEN), dtype = np.float32)
        for i, datum in enumerate(data_tuple):
            img_path, numbers, do_aug = datum
            label_seq[i, :len(numbers)] = numbers

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            #img = img[..., np.newaxis]
            #img = img/255.
            #img = np.transpose(img, (2, 0, 1))
            img_data[i] = img
            if (i+1) % 1000 == 0:
                print '[+] ###{} name: {}'.format(i, img_path)
                print '[+] number: {}'.format(','.join(map(lambda x: str(x), numbers)))
                print '[+] label: {}'.format(','.join(map(lambda x: str(x), label_seq[i])))
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('data', data = img_data)
            f.create_dataset('label', data = label_seq)
            print '=== H5 data written to ', file_name
    """

    with open(file_name, 'w') as f:
        workspace = os.path.split(file_name)[0]
        process_pool = []
        for g in xrange(groups):
            h5_file_name = os.path.join(workspace, '%s_%d.h5' %(phase, g))
            f.write(h5_file_name + '\n')
            start_idx = g*single_size
            end_idx = start_idx + single_size
            if g == groups - 1:
                end_idx = len(data_tuple)
            p = Process(target = process, args = (h5_file_name, data_tuple[start_idx:end_idx]))
            p.start()
            process_pool.append(p)
        for p in process_pool:
            p.join()


def batch_load(json_imgfolder_map, list_filename, phase):
    img = []
    pts = []
    for json_file, img_folder in json_imgfolder_map.iteritems():
        img_names = glob.glob(os.path.join(img_folder, '*.jpg'))
        print img_folder, '\tNumber of images=', len(img_names)
        json_file = json.load(open(json_file))
        for idx, i in enumerate(img_names):
            # pdb.set_trace()
            if os.path.basename(i) in json_file:
                pt = json_file[os.path.basename(i)]
                if len(pt) > 0:
                    found = False
                    for jj in range(len(pt)):
                        if pt[jj]['text'][0] not in exclude_chars:
                            china_plate_index = jj
                            found = True
                            break
                    if idx%1000==0 and found:
                        print idx, pt[china_plate_index]['text'].encode('utf8')
                    if found:
                        img.append(i)
                        pts.append(pt[china_plate_index]['coordinates'][:8])

    data_all = list(zip(img, pts))
    random.shuffle(data_all)
    write_image_info_into_hdf5(list_filename, data_all, phase)

    """
    X, y = load(img, pts)
    X = np.swapaxes(X, 2, 3)
    X = np.swapaxes(X, 1, 2)
    print(X.shape)

    #sep = int(y.shape[0] * .9)
    sep = int(y.shape[0] * 1.0)
    print(sep)
    #writeHdf5('train',X[0:sep],y[0:sep], sys.argv[1])
    #writeHdf5('val',X[sep:],y[sep:], sys.argv[1])
    writeHdf5Batch('train',X[0:sep],y[0:sep], sys.argv[1])
    #writeHdf5Batch('val',X[sep:],y[sep:], sys.argv[1])
    #writeHdf5('train',X,y, sys.argv[1])
    """

def parse_args():
    parser = argparse.ArgumentParser(description='Convert the labeling json files into h5 files for caffe corner training')
    parser.add_argument('--h5_path', default='/mnt/soulfs2/wfei/code/crnn.caffe/data/plate/crnn',
                        type=str, help='Path to write the h5 file and list file')
    parser.add_argument('--prefix', default='train', type=str, help='h5 file prefix')
    parser.add_argument('--list_name', default='plate_trainning_aug.list', type=str, help='list filename containing the list of h5 files')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    """k11 training data """ 
    json_imgfolder_map = {
        '/mnt/soulfs2/fzhou/data/wanda/label/20181119_carplate_wanda_0921.json':"/ssd/zq/parkinglot_pipeline/carplate/crops_all/data/WANDA/20180921/crops/",
        '/ssd/wfei/data/plate_for_label/k11_1003/20181101_carplate_k11_1003.json':'/ssd/wfei/data/plate_for_label/k11_1003/crops',
        '/ssd/wfei/data/plate_for_label/hk_double/20190515_HK_Double_Plates.json':'/ssd/wfei/data/plate_for_label/hk_double/car_crop_20190515',
        '/ssd/wfei/data/plate_for_label/hongkong/k11_car_crop/20190220_k11b_hongkong_cnplate_labels_all.json':'/ssd/wfei/data/plate_for_label/hongkong/k11_car_crop/all_crop/'
    }
    """wanda training data """ 
    json_imgfolder_map = {
        #'/mnt/soulfs2/fzhou/data/wanda/label/20181119_carplate_wanda_0921.json': "/ssd/zq/parkinglot_pipeline/carplate/crops_all/data/WANDA/20180921/crops/",
        '/ssd/wfei/data/plate_for_label/wanda_10k/20190311_wanda_1w_plate_det_benchmark_fixed.json': '/ssd/wfei/data/plate_for_label/wanda_10k/wanda_10k_filtered',
        '/ssd/wfei/data/plate_for_label/wanda_b1_may/20190523_wanda_b1_20190510_11_carcrop.json': '/ssd/wfei/data/plate_for_label/wanda_b1_may/car_crop',
        '/ssd/wfei/data/plate_for_label/wanda_entrance/20190604_wanda_entrance_may2829.json': '/ssd/wfei/data/plate_for_label/wanda_entrance/images',
        '/ssd/wfei/data/plate_for_label/wanda_generic_15k/20190531_wanda_generic_15k.json': '/ssd/wfei/data/plate_for_label/wanda_generic_15k/images',
        '/ssd/wfei/data/plate_for_label/wanda_track_bm/20190711_wanda_track_bm_20190707.json': '/ssd/wfei/data/plate_for_label/wanda_track_bm/20190707',
        '/ssd/wfei/data/plate_for_label/energy_wanda/20190601_energy_wanda_16k.json': '/ssd/wfei/data/plate_for_label/energy_wanda/images'
    }
    args = parse_args()
    if not os.path.exists(args.h5_path):
        os.makedirs(args.h5_path)
    list_file_name = os.path.join(args.h5_path, args.list_name)
    batch_load(json_imgfolder_map, args.h5_path, list_file_name, args.prefix)
