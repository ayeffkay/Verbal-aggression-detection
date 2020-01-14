"""Deep NN"""

from keras.models import Model
from keras.layers import Input, Dense, Embedding, \
    LSTM, MaxPooling1D, Conv1D
from keras.layers.core import SpatialDropout1D, Dropout, \
    Flatten


def load_embedding_layer(shape, vocab_size=None, embedding_dim=None,
                         embedding_matrix=None):
    if embedding_matrix is not None:
        embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=shape,
                                trainable=False)
    else:
        embedding_layer = Embedding(input_dim=vocab_size + 1,
                                    output_dim=embedding_dim,
                                    input_length=shape)
    return embedding_layer


def lstm(shape, lstm_units=128, spatial_dropout=0.2,
         embedding=False, vocab_size=None,
         embedding_dim=None, embedding_matrix=None):
    if embedding:
        inputs = Input(shape=(shape,))
        x = load_embedding_layer(shape, vocab_size, embedding_dim,
                                 embedding_matrix)(inputs)
    else:
        inputs = Input(shape=(1, shape))
        x = inputs
    x = SpatialDropout1D(spatial_dropout)(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Flatten()(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def stacked_lstm(shape, lstm_units=32, embedding=False, vocab_size=None,
         embedding_dim=None, embedding_matrix=None):
    if embedding:
        inputs = Input(shape=(shape,))
        x = load_embedding_layer(shape, vocab_size, embedding_dim,
                                 embedding_matrix)(inputs)
    else:
        inputs = Input(shape=(1, shape))
        x = inputs
    x = LSTM(lstm_units, return_sequences=True,
             input_shape=(1, shape))(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = LSTM(lstm_units)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    return model


def cnn(shape, nb_filters1=32, nb_filters2=64, kernel_size=3,
        pool_size=3, units=100, rate=0.2, embedding=False,
        vocab_size=None, embedding_dim=None, embedding_matrix=None):
    if embedding:
        inputs = Input(shape=(shape,))
        x = load_embedding_layer(shape, vocab_size, embedding_dim,
                                 embedding_matrix)(inputs)
    else:
        inputs = Input(shape=(shape, 1))
        x = inputs
    x = Conv1D(nb_filters1, kernel_size, activation="relu")(x)
    x = MaxPooling1D(pool_size)(x)
    x = Conv1D(nb_filters2, kernel_size, activation="relu")(x)
    x = MaxPooling1D(pool_size)(x)
    x = Flatten()(x)
    x = Dense(units, activation="relu")(x)
    x = Dropout(rate)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss="binary_crossentropy",
                  optimizer="adagrad",
                  metrics=["accuracy"])
    return model


def cnn_with_lstm(shape, embedding=False, vocab_size=None,
         embedding_dim=None, embedding_matrix=None):
    if embedding:
        inputs = Input(shape=(shape,))
        x = load_embedding_layer(shape, vocab_size, embedding_dim,
                                 embedding_matrix)(inputs)
    else:
        inputs = Input(shape=(shape, 1))
        x = inputs
    x = Dropout(0.2)(x)
    x = Conv1D(64, 3, padding='same', activation='relu', strides=1)(x)
    x = MaxPooling1D(2)(x)
    x = LSTM(70)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss="binary_crossentropy",
                  optimizer="adagrad",
                  metrics=["accuracy"])
    return model
