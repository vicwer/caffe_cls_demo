import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import cv2
import sys

def caffemodel_io(img_path, prototxt, model_path):
    img_size = 256
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = np.transpose(img, (2, 0, 1))
    img_mean = np.array([128.0, 128.0, 128.0], dtype=np.float32).reshape(3,1,1)
    img = (np.asarray(img) - img_mean)
    img = np.asarray(img) / 256.0

    deploy = prototxt
    model = model_path

    net = caffe.Net(deploy, model, caffe.TEST)
    net.blobs['data'].data[...] = img.reshape(1, 3, img_size, img_size)
    res = net.forward()
    print(res)

if __name__ == '__main__':
    img_path = sys.argv[1]
    print(img_path)
    prototxt = '../caffe_model/deploy.prototxt'
    model_path = '../caffe_model/cls_iter.caffemodel'

    caffemodel_io(img_path, prototxt, model_path)
