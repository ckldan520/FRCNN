# -*- coding: utf-8 -*-
import keras
from keras import layers
from model.RoiPoolingConv import RoiPoolingConv

#resblock  identity_block  unit
def identity_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    un_filter1, un_filter2, un_filter3 = filters

    x = layers.Conv2D(un_filter1, (1, 1),name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(axis=3, name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(un_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base+'2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(un_filter3, (1,1), name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base+'2c')(x)

    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    un_filter1, un_filter2, un_filter3 = filters

    x = layers.TimeDistributed(layers.Conv2D(un_filter1, (1, 1), kernel_initializer='normal'),
                               name=conv_name_base + '2a')(input_tensor)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(un_filter2, (kernel_size, kernel_size), padding='same', kernel_initializer='normal'), name=conv_name_base+'2b')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(un_filter3, (1,1), kernel_initializer='normal'), name=conv_name_base+'2c')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2c')(x)


    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x



#conv_block unit
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    un_filter1, un_filter2, un_filter3 = filters

    x = layers.Conv2D(un_filter1, (1, 1), strides=strides, name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(axis=3, name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(un_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base+'2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(un_filter3, (1,1), name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base+'2c')(x)

    shortcut = layers.Conv2D(un_filter3, (1, 1), strides=strides, name=conv_name_base+'1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=3, name=bn_name_base+'1')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape,strides=(2,2)):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    un_filter1, un_filter2, un_filter3 = filters

    x = layers.TimeDistributed(layers.Conv2D(un_filter1, (1, 1), strides=strides,
                                             kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(un_filter2, (kernel_size, kernel_size), padding='same', kernel_initializer='normal'), name=conv_name_base+'2b')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(un_filter3, (1,1), kernel_initializer='normal'), name=conv_name_base+'2c')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2c')(x)

    shortcut = layers.TimeDistributed(layers.Conv2D(un_filter3, (1, 1), strides=strides), name=conv_name_base+'1')(input_tensor)
    shortcut = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base+'1')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


#resblock total
def nn_base(args, input_tensor=None, trainable=False):
        n_dims = args.n_dims

        if input_tensor is None:
            ins_input = keras.Input(shape=(None, None, n_dims))
        else:
            if not keras.backend.is_keras_tensor(input_tensor):
                ins_input = keras.Input(tensor=input_tensor, shape=(None, None, n_dims))
            else:
                ins_input = input_tensor


        x= layers.ZeroPadding2D(padding=(3, 3))(ins_input)

        x = layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='random_uniform', name='conv1')(x)#下降两倍
        x = layers.BatchNormalization(axis= 3, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)  #下降两倍

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1,1) )
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a' )#下降两倍
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')  # 下降两倍
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        return x


#RPN network
def rpn(base_layers, num_anchors):
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = layers.Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = layers.Conv2D(num_anchors*4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]






#classifier layers
def classifier_layers(x, input_shape):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2,2))
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block= 'b')
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c')
    x = layers.TimeDistributed(layers.AveragePooling2D((7, 7), name='ava_pool'))(x)

    return x

#classifier network
def classifier(base_layers, input_rois, num_rois, nb_classes):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    #此处的roipoolingConv要求了一个batch仅有一个样本
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape)
    out = layers.TimeDistributed(layers.Flatten())(out)

    out_class = layers.TimeDistributed(layers.Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = layers.TimeDistributed(layers.Dense(4*(nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]