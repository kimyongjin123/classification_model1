import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from opencv import removeBackgroundFolder,singleRemoveBackground

from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense,Conv2D,