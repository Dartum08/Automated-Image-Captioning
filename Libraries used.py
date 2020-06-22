import tensorflow as tf
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import string
import os
from PIL import Image
import glob
from pickle import dump, load
import pickle
from time import time
import os

from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model, load_model, model_from_json
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.applications.vgg16 import VGG16


from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, TimeDistributed, RepeatVector, Reshape, concatenate, Dropout, BatchNormalization, Bidirectional
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import add


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical