#!/usr/bin/python
# coding: utf-8

# Imports
from sklearn.externals import joblib
import gzip # gzip is only needed if the file is compressed
import matplotlib.pyplot as plt

# Un-pickle de data
with gzip.open('dataset-training-set.pkl.gz', 'rb') as f:
    data, _ = joblib.load(f)

data_ = data

# Example to plot some images
plt.imshow(data_[0].reshape((48, 48)), cmap='gray')
plt.show()
