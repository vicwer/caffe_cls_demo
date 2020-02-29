# caffe_cls_demo


## Introduction
train image classfication network by caffe



This repo is organized as follows:

```
caffe_cls_demo/
    ├── data
        ├── imgs
        └── labels
    ├── README.md
    ├── caffe_pred.py
    ├── lmdb_io.py
    ├── deploy.prototxt
    └── solver.prototxt
```

## Requirements
1. caffe==1.0.0
2. python==3.6.8
3. numpy
4. opencv==3.4.1
5. Python packages might missing. pls fix it according to the error message.

## Installation, Prepare data, Training
### Installation
1. Clone the caffe_cls_caffe repository.

```
git clone https://github.com/vicwer/caffe_cls_caffe.git
```

2. Create data, lmdb and models directory. 

### Prepare data
data should be organized as follows:

```
data/
    |->labels/train.txt
    |->labels/test.txt
    |->imgs/*.png
```
1. Download dataset

2. Generate img_list.txt formatted as "img_path label"

3. Generate train.txt and test.txt formatted as "img_path label"

4. Generate lmdb:

```
python3 lmdb_io.py
```

### Train

```
caffe train --solver=deploy.prototxt -gpu 0
```

## Test:

```
python3 caffe_pred.py
```

## GOOD LUCK...
