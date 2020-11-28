import os
import logging
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, Flatten, BatchNormalization
from tensorflow.keras import backend as K
from util import *
from variables import *

np.random.seed(seed)
warnings.simplefilter("ignore", UserWarning)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class SVHNclassfication(object):
    def __init__(self):
        X, Y, Xtest, Ytest = load_data()
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.Ytest = Ytest
        print("X shape     : {}".format(self.X.shape))
        print("Xtest shape : {}".format(self.Xtest.shape))
        print("Y shape     : {}".format(self.Y.shape))
        print("Ytest shape : {}".format(self.Ytest.shape))

    def model_conversion(self):
        num_classes = len(set(self.Y))
        input_tensor = Input(shape=input_shape)
        functional_model = tf.keras.applications.MobileNetV2(
                                                    include_top=False,
                                                    weights="imagenet",
                                                    input_tensor=input_tensor
                                                             )
        functional_model.trainable = False

        inputs = functional_model.input
        x = functional_model.layers[-2].output
        x = Flatten()(x)
        x = Dense(dense1, activation='relu')(x)
        x = Dense(dense2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense4, activation='relu')(x)
        x = Dense(dense4, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(
                inputs =inputs,
                outputs=outputs,
                name='SVHNclassifier'
                    )
        self.model = model
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit(
                    self.X,
                    self.Y,
                    validation_split=val_split,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose
                        )

    def save_model(self):
        self.model.save(model_weights)
        print("Model Saved !!")

    def loading_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(model_weights)
        print("Model Loaded !!")

    def Evaluation(self):
        loss, accuracy = self.model.evaluate(self.Xtest, self.Ytest)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)

    def predictions(self, X):
        dims = len(X.shape)
        if dims == len(input_shape):
            X = np.array([X])
        P = self.model.predict(X)
        preds = P.argmax(axis=-1)
        return preds.squeeze()


    def run_MobileNet(self):
        self.model_conversion()
        self.train()
        self.Evaluation()

if __name__ == "__main__": 
    model = SVHNclassfication()
    model.run_MobileNet()