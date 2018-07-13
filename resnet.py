from keras.models import Model
from keras.layers import Dense, Input, Embedding
from keras.layers import GlobalMaxPool1D, Dropout, Conv1D, BatchNormalization, Activation, Add
from keras.utils import plot_model


def block(x, kernel_size):
    x_Conv_1 = Conv1D(filters=512, kernel_size=[kernel_size], strides=1, padding='same')(x)
    x_Conv_1 = Activation(activation='relu')(x_Conv_1)
    x_Conv_2 = Conv1D(filters=512, kernel_size=[kernel_size], strides=1, padding='same')(x_Conv_1)
    x_Conv_2 = Add()([x, x_Conv_2])
    x = Activation(activation='relu')(x_Conv_2)
    return x


if __name__ == '__main__':
    num_words = 80000
    maxlen = 400
    kernel_size = 3
    DIM = 512
    batch_size = 256

    data_input = Input(shape=[maxlen])
    word_vec = Embedding(input_dim=num_words + 1,
                         input_length=maxlen,
                         output_dim=DIM,
                         mask_zero=0,
                         name='Embedding')(data_input)
    block1 = block(x=word_vec, kernel_size=3)
    block2 = block(x=block1, kernel_size=3)
    x = GlobalMaxPool1D()(block2)
    x = BatchNormalization()(x)
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    plot_model(model, './resnet.png', show_shapes=True)
