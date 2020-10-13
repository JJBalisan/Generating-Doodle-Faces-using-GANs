import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import argmax
from numpy.random import randint
import random
import pickle
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

#Loads data and reshapes

def load_data(DataType = 0, DataSize = 100000):
    ##DataType is 0 for faces and 1 for planes
    #DataSize is size of output data frame

    root = "full_numpy_bitmap_"
    ext = ".npy"
    names = ["face", "airplane"]

    data = np.load(root + names[DataType] + ext)

    #Reshapes and normalizes Data
    data_reshaped = np.reshape(data, (data.shape[0], 28, 28, 1))/255

    #Returns Subset of data
    return data_reshaped[:DataSize-1]

#creates image greyscale numpy array
def test_image(image_array, filename = 'my.png', save = False, show = True):

    filename = "Images/" + filename

    image = image_array.reshape((28,28)) * 255

    img = Image.fromarray(image , 'L')

    if save:
        img.save(filename)

    if show:
        img.show()

def plot_accuracy(d_acc_history,a_acc_history, Run_Name = "test"):
    Title = "AccuracybyStep_" + Run_Name + "run.png"
    plt.figure(1)
    line1 = plt.plot(d_acc_history, label="discriminator accuracy")
    line2 = plt.plot(a_acc_history, label="adversarial accuracy")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.savefig(Title)

def plot_loss(d_loss_history,a_loss_history, Run_Name = "test"):
    Title = "LossbyStep" + Run_Name + "run.png"
    plt.figure(2)
    line1 = plt.plot(d_loss_history, label="discriminator loss")
    line2 = plt.plot(a_loss_history, label="adversarial loss")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.savefig(Title)

### Following code is from Rowel Atienza

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )
