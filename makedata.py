#!/usr/bin/python
# -*- coding: utf-8 -*-

#joblib is usually significantly faster on large numpy arrays because it
# has a special handling for the array buffers of the numpy datastructure

# Imports
from dateP.dateP import DateP
from sklearn.externals import joblib
import cv2
import gzip
import numpy as np
import os
import pickle
import random
import shutil

# text.txt or train.txt file path, should have 'imagepath label' as each row in .txt
file = 'dataset/train.txt'

# train or test and the total size of one image (48*48)
Object = DateP(file, 2304)
O1 = Object.date_process()
d = O1

# Print message
print('writing...\n')

# os.path.join([local], [filename])
filename = os.path.join('/home/alexandra', 'dataset-training-set.pkl')

# Generate the pkl file, save as .gz
# Dumping in a gzip compressed file using a compress level of 3
file = joblib.dump(d, filename + '.gz', compress=('gzip', 3))

print('Done!\n')