import os
import pickle
#import hickle
import csv
from PIL import Image
import math

import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from sklearn.utils import shuffle

print('All modules imported.')

# 1 LOAD DATA --------------------------------------------

# Load pickled data
training_file = "./101train_data.pkl"
validation_file= "./101val_data.pkl"
testing_file = "./101test_data.pkl"
#training_file = "./32x32pickle/train.p"
#validation_file= "./32x32pickle/valid.p"
#testing_file = "./32x32pickle/test.p"
signname_file = "./signnames.csv"



with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(signname_file) as f:
    f.readline() # skip the headers
    signnames = [row[1] for row in csv.reader(f)]

# The images' pixels in [0, 255]
X_train, y_train = train['features'], train['labels']
print("shape train features: ", np.shape(X_train))
print('features type:',type(X_train))
print("train type:", type(train))
print("labels type: ",type(y_train))
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# 3 preprocess the Data Set ------------------

def normalize(image_set):
    
    return image_set.astype(np.float32) / 128. - 1.

# RGB2YCrCb?
def rgb2gray(image_set):
    new_set = np.array([])
    for img in image_set:
        np.append(new_set, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return new_set

X_train =  normalize(X_train)
X_valid =  normalize(X_valid)
X_test =  normalize(X_test)

# X_train = rgb2gray(X_train)
# X_valid = rgb2gray(X_valid)
# X_test = rgb2gray(X_test)

print(X_train.shape, X_train.dtype)
print(X_valid.shape, X_valid.dtype)
print(X_test.shape, X_test.dtype)


# Save the data for easy access
#pickle_file = './150x150pre.pickle'
pickle_file = './101pre.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_features': X_train,
                    'train_labels': y_train,
                    'valid_features': X_valid,
                    'valid_labels': y_valid,
                    'test_features': X_test,
                    'test_labels': y_test,
                    #'signnames': signnames,
                },
                pfile, protocol=2)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')