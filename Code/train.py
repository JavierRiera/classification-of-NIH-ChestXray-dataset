import numpy as np
import sys
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model, load_model
from itertools import chain

from Code.ResNet import resnet_def


def dataset_reading(plot=True):
    """Reads csv file containing the information of the input, one-hot encodes the labels in a pandas dataframe and 
    divides the dataframe into train and test according to the Chest-Xray framework, avoiding data leakage.
    You can also plot the numebr of labels in each dataset.


    Args:
    plot(bool) -- set to True to plot the label distribution on each subsets

    Returns:
    train_df(dataframe) -- dataframe for the train subset
    test_df(dataframe) -- dataframe for the test/validation subset
    all_labels(list) -- list of label names
    """

    #Read dataset
    dataframe = pd.read_csv('Dataset_toy\\Data_Entry_2017.csv')

    #One-hot encode the labels and add columns
    dataframe['Finding Labels'] = dataframe['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    all_labels = np.unique(list(chain(*dataframe['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))

    for label in all_labels:
        if len(label)>1:
            dataframe[label] = dataframe['Finding Labels'].map(lambda y: 1.0 if (label in y) else 0.0)

    #divide into train_val and test using txt files
    train_val_file = open('C:\\Users\\javie\\Documents\\Python Scripts\\ChestXray\\Dataset_toy\\train_val_list.txt', 'r')
    train_val_list = train_val_file.read().splitlines()
    test_file = open('C:\\Users\\javie\\Documents\\Python Scripts\\ChestXray\\Dataset_toy\\test_list.txt', 'r')
    test_list = test_file.read().splitlines()

    train_df = dataframe[dataframe['Image Index'].isin(train_val_list)]
    test_df = dataframe[dataframe['Image Index'].isin(test_list)]

    #Print the shape of both dataframes
    print('The shape of train: ', train_df.shape, 'Shape of test: ', test_df.shape)

    #Represent the number of labels present in the dataset
    if plot:
        #complete dataset
        fig, ax1 = plt.subplots(figsize = (5, 5))
        ax1.bar(np.arange(len(all_labels)), np.sum(dataframe[all_labels].values, axis = 0))
        ax1.set_xticks(np.arange(len(all_labels)))
        ax1.set_xticklabels(all_labels, rotation = 90)
        ax1.set_title('Complete dataset')

        #Train dataset
        fig, ax2 = plt.subplots(figsize = (5, 5))
        ax2.bar(np.arange(len(all_labels)), np.sum(train_df[all_labels].values, axis = 0))
        ax2.set_xticks(np.arange(len(all_labels)))
        ax2.set_xticklabels(all_labels, rotation = 90)
        ax2.set_title('Train dataset')

        #Test dataset
        fig, ax3 = plt.subplots(figsize = (5, 5))
        ax3.bar(np.arange(len(all_labels)), np.sum(test_df[all_labels].values, axis = 0))
        ax3.set_xticks(np.arange(len(all_labels)))
        ax3.set_xticklabels(all_labels, rotation = 90)
        ax3.set_title('Test dataset')

        plt.show()
    return train_df, test_df, all_labels

def get_train_generator(df, image_dir, x_cols, y_cols, shuffle=True, batch_size=2, seed=42, target_w=512, target_h=512):
    """Preprocess training input data (normalization and data augmanetation) and creates a generator for the train subset.

    Args:
    df(dataframe) -- train dataframe
    image_dir(string) -- directory string to the data folder containing the images
    x_col(string) -- column in dataframe that contains the filenames
    y_cols(list or string) -- column/s in dataframe that has the target data
    shuffle(bool) -- shuffles the data
    batch_size(int) -- size of the batches of data
    seed(int) -- for reprodicibility
    target_w and target_h(int) -- size of the width and height respectivly of the generator images

    Returns:
    generator(DataFrameIterator)
    """

    #Normalize images
    image_generator = ImageDataGenerator(
        samplewise_center = True,
        samplewise_std_normalization = True,
        rotation_range=30, fill_mode='nearest',
        width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=True)
    
    # Flow from dataframe and directory
    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_cols,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w,target_h))

    return generator

def get_val_test_generator(val_df, test_df, train_df, image_dir, x_cols, y_cols, shuffle=True, batch_size=2, seed=42, target_w=512, target_h=512):
    """Normalizes test and validation input data and creates a generator for each subset.

    Args:
    val_df(dataframe) -- validation dataframe
    test_df(dataframe) -- test dataframe
    train_df(dataframe) -- train dataframe
    image_dir(string) -- directory string to the data folder containing the images
    x_col(string) -- column in dataframe that contains the filenames
    y_cols(list or string) -- column/s in dataframe that has the target data
    shuffle(bool) -- shuffles the data
    batch_size(int) -- size of the batches of data
    seed(int) -- for reprodicibility
    target_w and target_h(int) -- size of the width and height respectivly of the generator images

    Returns:
    valid_generator(DataFrameIterator)
    test_geerator(DataFrameIterator)
    """
    
    #Generator to sample dataset (we need to normalice based on the train set)
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe = train_df,
        directory= image_dir,
        x_col = 'Image Index',
        y_col = y_cols,
        class_mode= "raw",
        batch_size = batch_size,
        shuffle= shuffle,
        target_size=(target_w, target_h))

    #Get Data Sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    #Use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        samplewise_center = True,
        samplewise_std_normalization = True)

    image_generator.fit(data_sample)

    #Test and valid generators
    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_cols,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_cols,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w,target_h))

    return valid_generator, test_generator

def compute_class_freq(labels):
    """Calculates the frequency for each of the labels in the train dataset.

    Args:
    labels(array) -- matrix containing all the labels present in the train generator

    Returns:
    positive_freq(array) -- contains the percentual value of present positive labels in the generator. Float64 values from 0 to 1.
    negative_freq(array) -- contains the percentual value of present negative labels in the generator. Float64 values from 0 to 1.
    """
    N = labels.shape[0]
    positive_freq = (1/N)*np.sum(labels==1, axis=0)
    negative_freq = (1/N)*np.sum(labels==0, axis=0)

    return positive_freq, negative_freq

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """Defines the loss function.

    Args:
    pos_weights(array) -- percentual value of present positive labels in the generator. Float64 values from 0 to 1.
    neg_weights(array) -- percentual value of present negative labels in the generator. Float64 values from 0 to 1.

    Returns:
    weighted_loss(function)
    """

    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += -1*K.mean(pos_weights[i]*y_true[:,i]     * K.log(y_pred[:,i]+epsilon) + 
                              neg_weights[i]*(1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon))  
        return loss
    return weighted_loss

def compile_resnet_model(model, labels, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)):
    """initialices the loss function and compiles the model. metrics are Accuracy and AUC

    Args:
    model(object) -- model to compile
    labels(array) -- matrix containing all the labels present in the train generator
    optimizer(string or optimizer instance) -- see tf.keras.optimizers for different posibilities

    Returns:
    model(object) -- compiled model
    """

    pos_w, neg_w = compute_class_freq(labels)
    model.compile(optimizer = optimizer, loss = get_weighted_loss(pos_w, neg_w), metrics = [tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC()])
    return model

if __name__ == '__main__':

    train_df, test_df, all_labels = dataset_reading(plot=False)
    valid_df = test_df[:2000] #Validation dataframe wont be used, this code is made for cases we use a specific validation df
    test_df = test_df[2000:]


    image_dir = '.\\Dataset_toy\\images'
    train_generator = get_train_generator(train_df, image_dir, 'Image Index', all_labels)
    valid_generator, test_generator = get_val_test_generator(valid_df, test_df, train_df, IMAGE_DIR, 'Image Index', all_labels)

    x, y = train_generator.__getitem__(0)

    Training = True
    if Training:
        model = resnet_def(input_shape = x.shape[1:], output_size=train_generator.labels.shape[1])
        model = compile_resnet_model(model, labels=train_generator.labels, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9))
        callbacks = [
                     tf.keras.callbacks.EarlyStopping(patience=10),
                     tf.keras.callbacks.ModelCheckpoint(filepath='Models_saves\\ResNet34\\model.{epoch:02d}-{val_loss:.2f}.tf'),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, patience=4, verbose=1),
                     tf.keras.callbacks.TensorBoard(log_dir="C:\\Users\\javie\\Documents\\Python Scripts\\ChestXray\\Logs\\run9")
                    ]

        with tf.device('/gpu:0'):
            history = model.fit(train_generator,
                                validation_data=valid_generator,
                                epochs=15,
                                steps_per_epoch=3000,
                                callbacks=callbacks)

            # model.save_weights('Models_saves\ResNet34', overwrite = True)

            # print(history.history.keys())
            verbose = False

            if verbose:
                plt.figure(1)
                plt.plot(history.history['loss'], color = 'b', label='train')
                plt.plot(history.history['val_loss'], color = 'g', label='validation')
                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.title("Loss Curves")

                plt.figure(2)
                plt.plot(history.history['auc'], color = 'b', label='train')
                plt.plot(history.history['val_auc'], color = 'g', label='validation')
                plt.ylabel("AUC")
                plt.xlabel("epoch")
                plt.title("AUC Curves")

                plt.figure(3)
                plt.plot(history.history['accuracy'], color = 'b', label='train')
                plt.plot(history.history['val_accuracy'], color = 'g', label='validation')
                plt.ylabel("Accuracy")
                plt.xlabel("epoch")
                plt.title("Accuracy Curves")
                plt.show()

    else:
        model = ResNet34_def(input_shape=x.shape[1:], output_size=train_generator.labels.shape[1])
        model = compile_resnet_model(model, labels=train_generator.labels, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9))

        model.load_weights('Models_saves\ResNet34')
        

    results = model.predict(test_generator, verbose=1)
    print('resultados: shape {}, max {}, min {}'.format(results.shape, np.max(results), np.min(results)))

    diff = test_generator.labels - results
    print('diff: shape {}, max {}, min {}'.format(diff.shape, np.max(diff), np.min(diff)))

    eval_results = model.evaluate(test_generator, verbose = 1)
    print('resultados: test_loss = {}, test_accuracy = {}'.format(eval_results[0], eval_results[1]))
