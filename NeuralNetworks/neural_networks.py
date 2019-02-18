from keras import Sequential
from keras.layers import SeparableConv1D, Dropout, Embedding, MaxPooling1D, Dense, Flatten, AveragePooling1D, Conv2D, \
    MaxPooling2D, Activation


def separableCNN(num_features, shape):
    print(shape)
    model = Sequential()
    model.add(Embedding(input_shape=shape, input_dim=num_features, output_dim=200))
    print(model.output_shape)
    model.add(Dropout(rate=0.2))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(SeparableConv1D(filters=64, kernel_size=(3), activation='relu'))
    print(model.output_shape)
    model.add(AveragePooling1D())
    model.add(Flatten())
    model.add(Dense(1, activation='softmax'))

    return model

def cnn(num_features, shape):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=shape))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    model.add(Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    return model

def dense(num_features):
    model = Sequential()
    model.add(Dense(512, input_shape=(num_features,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model
