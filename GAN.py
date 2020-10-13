import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Reshape, BatchNormalization, UpSampling2D
from keras.layers import Conv2DTranspose, LeakyReLU, Activation
import matplotlib.pyplot as plt
from numpy import argmax
from numpy.random import randint
from keras.optimizers import Adam, RMSprop
import random
import pickle

#Loads data and reshapes
def load_dataset(filename):

    data = np.load(filename)

    data_shape = data.shape

    data_reshaped = np.reshape(data, (data_shape[0], 28, 28, 1))

    return data_reshaped

def load_data():

    try:
        x = pickle.load(open("Dataset", "rb"))
        y = pickle.load(open("Targets", "rb"))

    except (OSError, IOError) as e:
        x, y = make_dataset()
        pickle.dump(x,open("Dataset","wb"))
        pickle.dump(y,open("Targets","wb"))

    return x,y

#makes our dataset
def make_dataset():

    root = "Data/full_numpy_bitmap_"
    ext = ".npy"
    filenames = ["airplane","bandage","blackberry","carrot", "dresser", "penguin",
        "golf club", "hockey stick", "kangaroo", "lantern", "map", "moustache",
        "nail", "parachute", "raccoon", "shovel", "string bean", "truck", "yoga"
        ,"zigzag", "face"]

    targets = [0] * 5000 + [1] * 5000
    images = []

    for name in filenames:

        data = load_dataset(root + name + ext)

        if name == "face":

            data = data[:5000]
            for image in data:
                images.append(image)

        else:
            data = data[:250]
            for image in data:
                images.append(image)

    tmp = list(zip(images, targets))
    random.shuffle(tmp)
    images, targets = zip(*tmp)

    targets = np.array(targets)
    images  = np.array(images)

    return images, targets

def split_data(x,y):

    data_length = len(y)
    cutoff = round(data_length * 0.8)

    x_train = x[:cutoff]
    y_train = y[:cutoff]

    x_test = x[cutoff:]
    y_test = y[cutoff:]

    return x_train,y_train,x_test,y_test


#classifier model
def build_classifier(dropout = 0):

    neural_net = Sequential()

    neural_net.add(Conv2D(64,(5,5),strides=(2, 2),input_shape=(28,28,1)))
    neural_net.add(LeakyReLU())

    neural_net.add(Conv2D(64,(2,2),strides=(2, 2)))
    neural_net.add(LeakyReLU())

    neural_net.add(Conv2D(64,(2,2),strides=(2, 2)))
    neural_net.add(LeakyReLU())

    neural_net.add(Flatten())

    neural_net.add(Dropout(dropout))
    neural_net.add(Dense(128, activation='sigmoid'))

    neural_net.add(Dropout(dropout))
    neural_net.add(Dense(128, activation='sigmoid'))
    neural_net.add(Dense(1, activation='sigmoid'))

    neural_net.summary()
    return neural_net

def build_generator(dropout = 0):

    # generator = Sequential()
    #
    # depth = 1
    # dim = 28
    #
    # generator.add(Dense(dim*dim*depth, input_dim=100))
    # generator.add(BatchNormalization(momentum=0.9))
    # generator.add(Activation('relu'))
    # generator.add(Reshape((dim, dim, depth)))
    # generator.add(Dropout(dropout))
    #
    #
    # #generator.add(UpSampling2D())
    # # generator.add(Conv2DTranspose(int(depth/2), 5,padding='same', activation="relu"))
    # # generator.add(BatchNormalization())
    #
    # #generator.add(UpSampling2D())
    # #generator.add(Conv2DTranspose(int(depth/4), 5, padding='same', activation="relu"))
    # #generator.add(BatchNormalization(momentum=0.9))
    #
    # #generator.add(Conv2DTranspose(int(depth/8), 5, padding='same', activation="relu"))
    # #generator.add(BatchNormalization(momentum=0.9))
    #
    # generator.add(Conv2DTranspose(1, 1, activation="sigmoid"))
    #
    # generator.summary()
    # return generator

    generator = Sequential()
    dropout = 0.4
    depth = 64+64+64+64
    dim = 7
    # In: 100
    # Out: dim x dim x depth
    generator.add(Dense(dim*dim*depth, input_dim=100))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(Reshape((dim, dim, depth)))
    generator.add(Dropout(dropout))
    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2
    generator.add(UpSampling2D())
    generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(UpSampling2D())
    generator.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
    generator.add(Conv2DTranspose(1, 5, padding='same'))
    generator.add(Activation('sigmoid'))
    generator.summary()
    return generator

def adversarial_model(classifier, generator):
    optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
    model = Sequential()
    model.add(generator)
    model.add(classifier)
    model.compile(optimizer= optimizer, loss="binary_crossentropy",
                           metrics=['accuracy'])

    return model

def classifier_model(dropout = 0):

    model = Sequential()
    model.add(build_classifier(dropout))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy",
                           metrics=['accuracy'])
    return model

def train(classifier, generator, adversarial, training_set, training_targets, train_steps = 20, batch_size = 32):

    for i in range(train_steps):
        subset_indices = np.random.randint(0, training_set.shape[0], size=batch_size)

        images_subset = training_set[subset_indices]
        targets_subset = training_targets[subset_indices]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        fakes = generator.predict(noise)

        inputs = np.concatenate((images_subset, fakes))
        targets = np.concatenate((targets_subset, np.zeros(batch_size)))
        class_loss = classifier.train_on_batch(inputs, targets)

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        targets = np.ones(batch_size)
        gen_loss = adversarial.train_on_batch(noise, targets)
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, class_loss[0], class_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, gen_loss[0], gen_loss[1])
        print(log_mesg)

#creates image greyscale numpy array
def test_image(image_array, filename = 'my.png', save = False, show = True):

    filename = "Images/" + filename

    image = image_array.reshape((28,28))

    img = Image.fromarray(image , 'L')

    if save:
        img.save(filename)

    if show:
        img.show()

def main():

    x, y = load_data()
    x_train,y_train,x_test,y_test = split_data(x,y)

    for i in range(10):
        name = str(i) + ".png"
        test_image(x[i], name, False, False)

    #construct the classifier
    classifier = classifier_model(0.3)

    #Train the model
    history = classifier.fit(x_train, y_train, verbose=1,
                         validation_data=(x_test, y_test),
                         epochs=10)

    #assess performance
    loss, accuracy = classifier.evaluate(x_test, y_test, verbose=0)
    print("accuracy: {}%".format(accuracy*100))

    #construct generator and adversarial model
    generator = build_generator(0.6)
    adversarial = adversarial_model(classifier, generator)
    #train(classifier, generator, adversarial,  x_train, y_train, 3)

    noise = np.random.uniform(-1.0, 1.0, size=[3, 100])
    fakes = generator.predict(noise)

    for i in range(3):
        name = "Untrainedgenerator" + str(i) + ".png"
        test_image(fakes[i], name, True, True)


if __name__ == '__main__':
    main()
