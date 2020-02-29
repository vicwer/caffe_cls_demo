import lmdb
import os
import cv2
import numpy as np
import caffe

def write_lmdb(txt_path, lmdb_path):
    db = lmdb.open(lmdb_path, map_size=int(1e12))

    f = open(txt_path, 'r')
    lines = f.readlines()
    with db.begin(write=True) as db_txn:
        index = 0
        for i in lines:
            img = cv2.imread(i.strip().split(' ')[0])
            try:
                h, w, _ = img.shape
            except:
                continue
            img = cv2.resize(img, (256, 256))
            img = np.transpose(img, (2,0,1))
            img_dat = caffe.io.array_to_datum(img)
            img_dat.label = int(i.strip().split(' ')[1])

            db_txn.put(('{:0>10d}'.format(index)).encode(), img_dat.SerializeToString())
            index += 1

def read_lmdb(imdb_path):
    lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for key,val in lmdb_cursor:
        datum.ParseFromString(val)
        label = int(datum.label)
        print(label)
        data_array = caffe.io.datum_to_array(datum)
        img = np.fromstring(data_array, dtype=np.uint8)

    lmdb_env.close()

if __name__ == "__main__":
    # train.lmdb
    txt_path = "./imgs/train.txt"
    lmdb_path = "./train_lmdb"
    write_lmdb(txt_path, lmdb_path)
    #read_lmdb(lmdb_path)

    # test.lmdb
    txt_path = "./imgs/test.txt"
    lmdb_path = "./test_lmdb"
    write_lmdb(txt_path, lmdb_path)
