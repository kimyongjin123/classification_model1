import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from matplotlib.pyplot import imshow
import random
from sklearn.utils.validation import validate_data
from sympy import shape
from sympy.tensor.array.arrayop import Flatten
from classification_model import getTrainData
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from opencv01 import removeBackgroundFolder,singleRemoveBackground
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense,Conv2D,Dropout,MaxPool2D,Flatten
data_sets = getTrainData(r"d:\imgs")
label_list = data_sets["label_list"]
x_train,y_train = data_sets["train"]
x_test,y_test = data_sets["test"]
y_test = tf.one_hot(y_test,10)
print(x_test.shape)
print(y_test.shape)
import pickle
with open("classification_image.history","rb") as fp:
    fit_his=pickle.load(fp)
print(fit_his.history.keys())
plt.subplot(1,2,2)
plt.plot(fit_his.history["loss"],label="Train Loss")
plt.plot(fit_his.history["val_loss"], label="Valid Loss")
plt.legend()
plt.title("LOSSES")
plt.show()
model= tf.keras.models.load_model("classification_image.keras")
loss,acc = model.evaluate(x_test,y_test)
print("손실값:",loss,"정확도:",acc)
y_pred = model.predict(x_test)
rindex = [random.randint(0,len(x_test)) for ix in range(10)]
for i in range(len(rindex)):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[rindex[i]])
    plt.title(label_list[np.argmax(y_test[rindex[i]])])
    plt.xlabel(f"predict : {label_list[np.argmax(y_pred[rindex[i]])]}")
    plt.xticks([]);plt.yticks([])
plt.show()
#혼동행렬
y_conv_true = np.array([np.argmax(ll) for ll in y_test])
y_conv_pred = np.array([np.argmax(ll) for ll in y_pred])
print(y_conv_true.shape)
print(y_conv_pred.shape)
print(y_conv_true[1:10])
print(y_conv_true[1:10])
#confusion_matrix()

