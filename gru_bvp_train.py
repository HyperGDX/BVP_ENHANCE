from __future__ import print_function

import os
import sys
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, TimeDistributed, BatchNormalization, SeparableConv2D
from keras.models import Model, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix
from keras.backend import set_session
from sklearn.model_selection import train_test_split
from cvae_bvp_train import LATENT_DIM
from model import cvae_tf

#### Parameters ####
use_existing_model = False
# train:test = 9:1
fraction_for_test = 0.1
# 带有递归
data_dir = 'testdata/user1'
ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)
T_MAX = 0
n_epochs = 100
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 64
f_learning_rate = 0.001


def normalize_data(data_1):
    # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
    data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
    return data_1_norm


def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values=0).tolist())
    return np.array(data_pad)


def onehot_encoding(label, num_class):
    # label(list)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]     # from label to onehot
    return _label


def load_data(path_to_data, motion_sel):
    global T_MAX
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root, data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])
                location = int(data_file_name.split('-')[2])
                orientation = int(data_file_name.split('-')[3])
                repetition = int(data_file_name.split('-')[4])

                # Select Motion
                if (label_1 not in motion_sel):
                    continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue

                # Normalization
                data_normed_1 = normalize_data(data_1)

                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]
            except Exception:
                continue

            # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1)

    # Zero-padding
    data = zero_padding(data, T_MAX)

    # Swap axes
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
    data = np.expand_dims(data, axis=-1)    # [N,T_MAX,20,20]=>[N,T_MAX,20,20,1]

    # Convert label to ndarray
    label = np.array(label)

    # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
    return data, label


def assemble_model(input_shape, n_class):
    # input_shape (T_MAX, 12)
    model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')
    x = GRU(n_gru_hidden_units, return_sequences=False)(model_input)
    model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)

    # Model compiling
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model
# def assemble_model(input_shape, n_class):
#     # input_shape (T_MAX, 20, 20, 1)
#     model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')    # (@,T_MAX,20,20,1)

#     # Feature extraction part

#     x = TimeDistributed(SeparableConv2D(16, kernel_size=(5, 5), activation='relu', data_format='channels_last',
#                                         input_shape=input_shape))(model_input)   # (@,T_MAX,20,20,1)=>(@,T_MAX,16,16,16)
#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)    # (@,T_MAX,16,16,16)=>(@,T_MAX,8,8,16)
#     x = TimeDistributed(Flatten())(x)   # (@,T_MAX,8,8,16)=>(@,T_MAX,8*8*16)
#     x = TimeDistributed(Dense(64, activation='relu'))(x)  # (@,T_MAX,8*8*16)=>(@,T_MAX,64)
#     x = TimeDistributed(Dropout(f_dropout_ratio))(x)
#     x = TimeDistributed(Dense(64, activation='relu'))(x)  # (@,T_MAX,64)=>(@,T_MAX,64)
#     x = GRU(n_gru_hidden_units, return_sequences=False)(x)  # (@,T_MAX,64)=>(@,128)
#     x = Dropout(f_dropout_ratio)(x)
#     model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)  # (@,128)=>(@,n_class)

#     # Model compiling
#     model = Model(inputs=model_input, outputs=model_output)
#     model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy']
#                   )
#     return model


# def assemble_model(input_shape, n_class):
#     # input_shape (T_MAX, 20, 20, 1)
#     model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')    # (@,T_MAX,20,20,1)

#     # Feature extraction part

#     x = TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation='relu', data_format='channels_last',
#                                input_shape=input_shape))(model_input)   # (@,T_MAX,20,20,1)=>(@,T_MAX,18,18,16)

#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)    # (@,T_MAX,18,18,16)=>(@,T_MAX,8,8,16)
#     x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', data_format='channels_last',
#                                input_shape=[input_shape[0], 18, 18, 16]))(model_input)   # (@,T_MAX,18,18,16)=>(@,T_MAX,16,16,32)
#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)    # (@,T_MAX,16,16,32)=>(@,T_MAX,8,8,32)
#     x = TimeDistributed(Flatten())(x)   # (@,T_MAX,8,8,32)=>(@,T_MAX,8*8*32=2048)
#     x = TimeDistributed(Dense(1024, activation='relu'))(x)  # (@,T_MAX,2048)=>(@,T_MAX,1024)
#     x = TimeDistributed(Dropout(f_dropout_ratio))(x)
#     x = TimeDistributed(Dense(512, activation='relu'))(x)  # (@,T_MAX,1024)=>(@,T_MAX,512)
#     x = TimeDistributed(Dropout(f_dropout_ratio))(x)
#     x = TimeDistributed(Dense(64, activation='relu'))(x)  # (@,T_MAX,512)=>(@,T_MAX,64)
#     x = GRU(n_gru_hidden_units, return_sequences=False)(x)  # (@,T_MAX,64)=>(@,128)
#     x = Dropout(f_dropout_ratio)(x)
#     model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)  # (@,128)=>(@,n_class)

#     # Model compiling
#     model = Model(inputs=model_input, outputs=model_output)
#     model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy']
#                   )
#     return model


# ==============================================================
# Let's BEGIN >>>>

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))
tf.random.set_seed(1)


# Load data
data, label = load_data(data_dir, ALL_MOTION)

cvae_model = cvae_tf.CVAE(latent_dim=LATENT_DIM, T_MAX=T_MAX)
tf.saved_model.load("model/cvae_model_save.pt")
data_cvae = cvae_model.encoder(data).numpy()

print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0, :, :].shape) + '\n')

# Split train and test
[data_train, data_test, label_train, label_test] = train_test_split(data_cvae, label, test_size=fraction_for_test)
print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +
      'Test on ' + str(label_test.shape[0]) + ' samples\n')

# One-hot encoding for train data
label_train = onehot_encoding(label_train, N_MOTION)

# Load or fabricate model
if use_existing_model:
    model = load_model('model_widar3_trained.h5')
    model.summary()
else:
    model = assemble_model(input_shape=(T_MAX, 64), n_class=N_MOTION)
    model.summary()
    model.fit({'name_model_input': data_train}, {'name_model_output': label_train},
              batch_size=n_batch_size,
              epochs=n_epochs,
              verbose=1,
              validation_split=0.1, shuffle=True)
    print('Saving trained model...')
    model.save('model_widar3_trained.h5')

# Testing...
print('Testing...')
label_test_pred = model.predict(data_test)
label_test_pred = np.argmax(label_test_pred, axis=-1) + 1

# Confusion Matrix
cm = confusion_matrix(label_test, label_test_pred)
print(cm)
cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm = np.around(cm, decimals=2)
print(cm)

# Accuracy
test_accuracy = np.sum(label_test == label_test_pred) / (label_test.shape[0])
print(test_accuracy)
