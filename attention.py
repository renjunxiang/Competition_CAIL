from keras.layers import *
from keras.models import *
from keras.utils import plot_model


def attention(input=None, depth=None):
    attention = Dense(1, activation='tanh')(input)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(depth)(attention)
    attention = Permute([2, 1], name='attention_vec')(attention)
    attention_mul = Multiply(name='attention_mul')([input, attention])
    return attention_mul


if __name__ == '__main__':
    data_input = Input(shape=[400])
    word_vec = Embedding(input_dim=40000 + 1,
                         input_length=400,
                         output_dim=512,
                         mask_zero=False,
                         name='Embedding')(data_input)
    x = word_vec
    x = Conv1D(filters=512, kernel_size=[3], strides=1, padding='same', activation='relu')(x)
    x = attention(input=x, depth=512)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(500, activation="relu")(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    plot_model(model, './attention.png', show_shapes=True)
