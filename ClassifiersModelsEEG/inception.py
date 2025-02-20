
"""
@author: Steven Cao"""

#IMPORT ALL NEEDED MODULES

#Standard library imports
import datetime
import numpy as np
import time

#Third party imports
#from keras.utils.vis_utils import plot_model
import tensorflow as tf
import tensorflow.keras as keras

#Local application imports
from utilsEEG import calculate_metrics
from utilsEEG import save_logs
from utilsEEG import save_test_duration


"""
The Inception Network
    Parameters:
        - output_directory (str): the str to the output directory which will contain the results.
        - input_shape (int,int) : the shape of each sample 
        - nb_classes (int)      : the number of classes 
        - verbose (boolean)     : whether or not you want to print out the training process.
        - batch_size (int)      : how many samples are you sending to the model before it updates its weights
        - lr (float)            : the learning rate of the model
        - nb_filters (int)      : how many filters for each convolutional layer
        - depth (int)           : how many inception blocks the Inception Network will have
        - kernel_size (int)     : the size of each kernel for each convolutional layer
        - nb_epochs (int)       : the number of epochs
    Return: none

"""

class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, lr=0.009,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=150): #nb_epochs=1500
        self.output_directory = output_directory
        self.nb_filters       = nb_filters
        self.use_residual     = use_residual
        self.use_bottleneck   = use_bottleneck
        self.depth            = depth
        self.kernel_size      = kernel_size - 1
        self.callbacks        = None
        self.batch_size       = batch_size
        self.bottleneck_size  = 32
        self.nb_epochs        = nb_epochs
        self.lr               = lr
        self.verbose          = verbose
        self.input_shape      = input_shape
        self.nb_classes       = nb_classes

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.h5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list     = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6     = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)
        x           = input_layer
        input_res   = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer    = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model        = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        #CREATING MODEL CALLBACKS
        reduce_lr            = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
        file_path            = self.output_directory + 'best_model.h5'
        model_checkpoint     = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
        log_dir              = self.output_directory + 'tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True, update_freq="epoch")
        self.callbacks       = [reduce_lr, model_checkpoint, tensorboard_callback]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.h5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        #SAVE PREDICTIONS
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        #CONVERT THE PREDICTED FROM BINARY TO INTEGER
        y_pred     = np.argmax(y_pred, axis=1)


        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)
        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):

        #LOADING THE MODEL
        start_time = time.time()
        model_path = self.output_directory + 'best_model.h5'
        model = keras.models.load_model(model_path)

        #PREDICTING THE SAMPLES FROM THE TEST DATASET
        time1  = time.time()
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        time2  = time.time()
        print("time to go through the test set:", time2 - time1)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
