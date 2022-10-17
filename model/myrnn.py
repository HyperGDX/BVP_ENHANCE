import tensorflow as tf
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, TimeDistributed, BatchNormalization, SeparableConv2D
from keras.models import Model, load_model
import keras


def assemble_model(input_shape, n_gru_hidden_units, n_class, f_dropout_ratio=0.5, f_learning_rate=1e-2):
    # input_shape (T_MAX, 8)
    model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')    # (@,T_MAX,20,20,1)

    # Feature extraction part
    x = GRU(n_gru_hidden_units, return_sequences=False)(x)  # (@,T_MAX,64)=>(@,128)
    x = Dropout(f_dropout_ratio)(x)
    model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)  # (@,128)=>(@,n_class)

    # Model compiling
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model
