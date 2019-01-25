from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Convolution1D, Activation, GlobalMaxPooling1D, Dropout, Dense
from keras.optimizers import SGD


def CNN_Model(x_train, y_train, x_test, y_test):
    print("training")
    num_docs = len(x_train)
    num_words = len(x_train[0])

    inp = Input(shape=(num_words,))
    out = Embedding(input_dim=num_docs, output_dim=64, input_length=num_words, dropout=0.2)(inp)
    out = Convolution1D(256, 64, border_mode='same', subsample_length=1)(out)
    out = Activation('relu')(out)
    out = GlobalMaxPooling1D()(out)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(inp, out)

    best_weights_filepath = './best_multi_domain_weights.hdf5'
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                    mode='auto')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(x_test, y_test), verbose=1,
              callbacks=[saveBestModel])

    json_string = model.to_json()
    open('multi_domain_archi.json', 'w').write(json_string)
    model.save_weights('final_multi_domain_weights.h5')

    return model
