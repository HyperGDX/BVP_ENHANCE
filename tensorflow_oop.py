import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Dense
import numpy as np
import time

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0


class Mnist(tf.keras.Model):
    def __init__(self):
        super(Mnist, self).__init__()
        self.dense1 = Flatten(input_shape=(28, 28))
        self.dense2 = Dense(100, activation='relu')
        self.dense3 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        return super().compute_loss(x, y, y_pred, sample_weight)

    def compute_metrics(self, x, y, y_pred, sample_weight):
        return super().compute_metrics(x, y, y_pred, sample_weight)


model = Mnist()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
start_time = time.time()

model.fit(train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_freq=1)
print(model.metrics_names)
