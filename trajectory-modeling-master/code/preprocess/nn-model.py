from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import keras.backend as K
from keras.callbacks import LambdaCallback


def CON_BLOCK(INPUT, n_filters, kernel_size=3, batchnorm=True, name=None):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same", name=name)(INPUT)

    if batchnorm:
        x = BatchNormalization()(x)

    #     x = LeakyReLU(alpha=0.2)(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #     x = LeakyReLU(alpha=0.2)(x)

    return x


def unet(INPUT, n_filters=64, dropout=0.5, batchnorm=True):
    # Downside path
    c1 = CON_BLOCK(INPUT, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, name='conv_1')
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = CON_BLOCK(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, name='conv_2')
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = CON_BLOCK(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, name='conv_3')
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = CON_BLOCK(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, name='conv_4')
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = CON_BLOCK(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, name='conv_5')

    # Upside path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = CON_BLOCK(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, name='conv_6')

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = CON_BLOCK(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, name='conv_7')

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = CON_BLOCK(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, name='conv_8')

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = CON_BLOCK(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, name='conv_9')

    OUTPUT = Conv2D(3, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[INPUT], outputs=[OUTPUT])
    return model