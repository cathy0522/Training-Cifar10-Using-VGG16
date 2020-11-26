import tkinter
from tkinter import *
import os
import random
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import Model


folder_path = 'cifar-10-batches-py'
# parameters
batch_size = 32
lr = 0.001
optimizer = "SGD"
display_step = 10
TRAINING_STEPS = 6000

# networks parameters
data_input = 3072
classes = 10
dropout = 0.60

def load_train_data(folder_path, images_id):  # n = 1, 2, ..., 5, data_batch_1.bin until data_batch_5.bin
    """Load Cifar10 data from `path`"""
    images_path = os.path.join(folder_path, 'data_batch_{}'.format(images_id))
    with open(images_path, 'rb') as file:
        images = pickle.load(file, encoding='latin1')
    # print(len(images['data']))
    feature = images['data'].reshape((len(images['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    label = images['labels']
    # print(features, labels)
    return feature, label

def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1.0

    return encoded

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def Train_Images(features, labels):
    # download cifar10
    # if not isfile('cifar-10-python.tar.gz'):
    #         urlretrieve(
    #             'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    #             'cifar-10-python.tar.gz')
    #
    # if not isdir('cifar-10-batches-py'):
    #     with tarfile.open('cifar-10-python.tar.gz') as tar:
    #         tar.extractall()
    #         tar.close()

    sample_id = 50000

    label_names = load_label_names()

    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []

    for i in range(sample_id):
        if (labels[i] == 0):
            list0.append(features[i])
        elif (labels[i] == 1):
            list1.append(features[i])
        elif (labels[i] == 2):
            list2.append(features[i])
        elif (labels[i] == 3):
            list3.append(features[i])
        elif (labels[i] == 4):
            list4.append(features[i])
        elif (labels[i] == 5):
            list5.append(features[i])
        elif (labels[i] == 6):
            list6.append(features[i])
        elif (labels[i] == 7):
            list7.append(features[i])
        elif (labels[i] == 8):
            list8.append(features[i])
        elif (labels[i] == 9):
            list9.append(features[i])

    fig, axs = plt.subplots(2, 5, figsize=(10, 10))

    axs[0, 0].imshow(list0[random.randint(0, len(list0))])
    axs[0, 0].set_title(label_names[0])
    axs[0, 1].imshow(list1[random.randint(0, len(list1))])
    axs[0, 1].set_title(label_names[1])
    axs[0, 2].imshow(list2[random.randint(0, len(list2))])
    axs[0, 2].set_title(label_names[2])
    axs[0, 3].imshow(list3[random.randint(0, len(list3))])
    axs[0, 3].set_title(label_names[3])
    axs[0, 4].imshow(list4[random.randint(0, len(list4))])
    axs[0, 4].set_title(label_names[4])
    axs[1, 0].imshow(list5[random.randint(0, len(list5))])
    axs[1, 0].set_title(label_names[5])
    axs[1, 1].imshow(list6[random.randint(0, len(list6))])
    axs[1, 1].set_title(label_names[6])
    axs[1, 2].imshow(list7[random.randint(0, len(list7))])
    axs[1, 2].set_title(label_names[7])
    axs[1, 3].imshow(list8[random.randint(0, len(list8))])
    axs[1, 3].set_title(label_names[8])
    axs[1, 4].imshow(list9[random.randint(0, len(list9))])
    axs[1, 4].set_title(label_names[9])

    plt.show()


def Hyperparameters(batch_size, lr, optimizer):
    print("hyperparameters: ")
    print("batch size: ", batch_size)
    print("learning rate: ", lr)
    print("optimizer: ", optimizer)


def Model_Structure():
    Model.Model()

def Accuracy():
    acc = mpimg.imread('Accuracy.png')
    loss = mpimg.imread('Loss.png')

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(acc)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(loss)
    plt.axis('off')

    plt.show()

def load_test_data(folder_path):  # n = 1, 2, ..., 5, data_batch_1.bin until data_batch_5.bin
    """Load Cifar10 data from `path`"""
    images_path = os.path.join(folder_path, 'test_batch')
    with open(images_path, 'rb') as file:
        images = pickle.load(file, encoding='latin1')
    # print(len(images['data']))
    feature = images['data'].reshape((len(images['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    label = images['labels']
    # print(features, labels)
    return feature, label

def Index(spinbox_tk):
    return spinbox_tk.get()

def Test(features_y, labels_y):
    test_num = int(Index(spinbox_tk))
    label_names = load_label_names()

    # plt.title(label_names[num])
    plt.imshow(features_y[test_num])

    plt.figure(figsize=(12, 5))
    plt.bar(label_names, predict[test_num])  # (x, y)

    pylab.show()

if __name__ == '__main__':
    root = tkinter.Tk()
    root.title("GUI")
    root.geometry("300x400")

    # Load Cifar-10 train image and label
    features1, labels1 = load_train_data(folder_path, 1)
    features2, labels2 = load_train_data(folder_path, 2)
    features3, labels3 = load_train_data(folder_path, 3)
    features4, labels4 = load_train_data(folder_path, 4)
    features5, labels5 = load_train_data(folder_path, 5)

    features_x = np.concatenate((features1, features2, features3, features4, features5), axis=0)  # (50000, 32, 32, 3)
    labels_x = np.concatenate((labels1, labels2, labels3, labels4, labels5), axis=0)  # (50000,)
    train_labels = one_hot_encode(labels_x)  # train_lable.shape= (50000, 10)

    features_y, labels_y = load_test_data(folder_path)
    test_labels = one_hot_encode(labels_y)

    model = tf.keras.models.load_model('vgg16_model.h5', custom_objects=None, compile=True, options=None)
    predict = model.predict(features_y)
    predict_classes = model.predict_classes(features_y)

    btn1 = tkinter.Button(root, text="1. Show Train Images", height=3, width=30,
                          command=lambda: Train_Images(features_x, labels_x))
    btn1.pack(pady=6)
    btn2 = tkinter.Button(root, text="2. Show Hyperparameters", height=3, width=30,
                          command=lambda: Hyperparameters(batch_size, lr, optimizer))
    btn2.pack(pady=6)
    btn3 = tkinter.Button(root, text="3. Show Model Structure", height=3, width=30,
                          command=lambda: Model_Structure())
    btn3.pack(pady=6)
    btn4 = tkinter.Button(root, text="4. Show Accuracy", height=3, width=30,
                          command=lambda: Accuracy())
    btn4.pack(pady=6)
    spinbox_tk = Spinbox(root, from_=0, to=9999, width=30, command=lambda: Index(spinbox_tk))
    spinbox_tk.pack(ipady=5, pady=6)
    btn5 = tkinter.Button(root, text="5. Test", height=3, width=30,
                          command=lambda: Test(features_y, labels_y))
    btn5.pack(pady=6)

    root.mainloop()
