import numpy as np
import pandas as pd
import sys
import json
import os
import sklearn.metrics as metrics
import utility.wide_residual_network as wrn
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from utility.watermark_regularizers import WatermarkRegularizer
from utility.watermark_regularizers import get_wmark_regularizers
from utility.watermark_regularizers import show_encoded_wmark
from utility.show_h5_keys import hdf5_upd
from utility.main_functions import save_watermark_sign, schedule

# RTX 2060 configuration settings
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

RESULT_PATH = './result'
MODEL_CHKPOINT_FNAME = os.path.join(RESULT_PATH, 'WRN-Weights.h5')

if __name__ == '__main__':
    # Getting the settings from the JSON files inside the config folder
    settings_json_filename = sys.argv[1]
    train_settings = json.load(open(settings_json_filename))
    
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    # Loading the dataset and fitting data for learning
    # It is supported only the cifar10 dataset.
    # The caltech101 is too large to be used correctly
    if train_settings['dataset'] == 'cifar10':
        dataset = cifar10
        classes_num = 10
    else:
        print('Dataset "{}" is not supported. Change the json settings and try again.'.format(train_settings['dataset']))
        exit(1)

    # Preprocessing the images of the cifar10 dataset
    # using an ImageDataGenerator of keras.
    generator = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,
                                   horizontal_flip=True)
    
    (trainX, trainY), (testX, testY) = dataset.load_data()
    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0
    trainY = kutils.to_categorical(trainY)
    testY = kutils.to_categorical(testY)
    
    # Fitting the model inside the image generator
    generator.fit(trainX, seed=0, augment=True)

    # Read parameters from JSON settings
    batch_size = train_settings['batch_size']
    epoch_num = train_settings['epoch']
    scale = train_settings['scale']
    embed_dim = train_settings['embed_dim']
    N = train_settings['N']
    k = train_settings['k']
    target_blk_id = train_settings['target_blk_id']
    base_modelw_fname = train_settings['base_modelw_fname']
    wtype = train_settings['wmark_wtype']
    randseed = train_settings['randseed'] if 'randseed' in train_settings else 'none'
    history_filename = train_settings['history']
    history_hdf_path = 'WTYPE_{}/DIM{}/SCALE{}/N{}K{}B{}EPOCH{}/TBLK{}'.format(wtype, embed_dim, scale, N, k, batch_size, epoch_num, target_blk_id)
    modelname_prefix = os.path.join(RESULT_PATH, 'wrn_' + history_hdf_path.replace('/', '_'))

    # Initialize the process for the watermark
    # Creating the b vector which will be the watermark
    b = np.ones((1, embed_dim))
    watermark_regularizer = WatermarkRegularizer(scale, b, wtype=wtype, randseed=randseed)

    init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
    model = wrn.create_wide_residual_network(init_shape, nb_classes=classes_num, N=N, k=k, dropout=0.00, wmark_regularizer=watermark_regularizer, target_blk_num=target_blk_id)
    model.summary()
    print('Watermark matrix:\n{}'.format(watermark_regularizer.get_matrix()))

    # Starting the training process
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])
    if len(base_modelw_fname) > 0:
        model.load_weights(base_modelw_fname)
    print("Finished compiling.")

    # We tested the history with just 3 epochs and find out that the results weren't satisfying. We then increased the
    # number of epochs to 10, 25 and 50. The most satistying results arrived with 200 epochs!
    history = model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=epoch_num,
                        callbacks=[callbacks.ModelCheckpoint(MODEL_CHKPOINT_FNAME, monitor="val_acc", save_best_only=True),
                                   LearningRateScheduler(schedule=schedule)
                        ],
                        validation_data=(testX, testY),
                        nb_val_samples=testX.shape[0],)
    show_encoded_wmark(model)

    # Validate the training accuracy
    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yPred = kutils.to_categorical(yPred)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy

    #   3 EPOCHS: Accuracy: 63.49 - Error: 36.51 (batch:  64)
    #  10 EPOCHS: Accuracy: 69.82 - Error: 30.18 (batch: 128)
    #  25 EPOCHS: Accuracy: 82.00 - Error: 18.00 (batch: 128)
    #  50 EPOCHS: Accuracy: 85.75 - Error: 14.25 (batch: 128)
    # 200 EPOCHS: Accuracy: 92.11 - Error:  7.89 (batch:  64)
    print("Accuracy: ", accuracy)
    print("Error: ", error)

    # Write history and model parameters to the result folder
    hdf5_upd(history_filename, history_hdf_path, pd.DataFrame(history.history))
    model.save_weights(modelname_prefix + '.weight')

    # Write the watermark matrix and the embedded signature to the result folder
    if target_blk_id > 0:
        save_watermark_sign(modelname_prefix, model)