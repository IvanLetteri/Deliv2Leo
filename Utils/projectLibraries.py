import tensorflow as tf

#list of 4 scaler used
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
#list of 2 transformerused
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn import preprocessing

import io
import time
import datetime
import os
import IPython

import numpy as np
import pandas as pd
import pandas_profiling as pp

import matplotlib.pyplot as plt
import h5py

import keras

from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

from keras.models import Sequential
from keras.models import model_from_json

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
from keras.optimizers import Adam, SGD

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix

