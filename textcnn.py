from keras.layers import Conv1D, BatchNormalization, Activation, GlobalMaxPool1D


def textcnn_one(word_vec=None, kernel_size=1, filters=512):
    x = word_vec
    x = Conv1D(filters=filters, kernel_size=[kernel_size], strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv1D(filters=filters, kernel_size=[kernel_size], strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = GlobalMaxPool1D()(x)

    return x
if __name__ == '__main__':
    from keras.layers import Dense, Embedding, Input, Dropout
    from keras.layers import BatchNormalization, Concatenate
    from keras.models import Model
    from keras.utils import plot_model

    filters=256
    data_input = Input(shape=[400])
    word_vec = Embedding(input_dim=40000 + 1,
                         input_length=400,
                         output_dim=512,
                         mask_zero=False,
                         name='Embedding')(data_input)

    x1 = textcnn_one(word_vec=word_vec, kernel_size=1, filters=filters)
    x2 = textcnn_one(word_vec=word_vec, kernel_size=2, filters=filters)
    x3 = textcnn_one(word_vec=word_vec, kernel_size=3, filters=filters)
    x4 = textcnn_one(word_vec=word_vec, kernel_size=4, filters=filters)
    x5 = textcnn_one(word_vec=word_vec, kernel_size=5, filters=filters)

    x = Concatenate(axis=1)([x1, x2, x3, x4, x5])
    x = BatchNormalization()(x)
    x = Dense(500, activation="relu")(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    plot_model(model, './textcnn.png', show_shapes=True)
