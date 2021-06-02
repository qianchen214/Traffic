# coding:utf-8
from tensorflow.keras.layers import Layer, InputSpec
from keras import initializers, regularizers
from keras import backend as K
from keras.layers import BatchNormalization
import torch
import torch.nn as nn

class FixedBatchNormalization(Layer):
    '''
        # 此处使用的是参数固定的BN层，参数在ImageNet是进行了训练，在微调网络的时候固定不动，当成了线性单元
        copy form
        https://github.com/yhenon/keras-frcnn/issues/33
        For the usage of BN layers, after pre-training, we com-
    pute the BN statistics (means and variances) for each layer
    on the ImageNet training set. Then the BN layers are fixed
    during fine-tuning for object detection. As such, the BN
    layers become linear activations with constant offsets and
    scales, and BN statistics are not updated by fine-tuning. We
    fix the BN layers mainly for reducing memory consumption
    in Faster R-CNN training.
    '''

    def __init__(self, epsilon=1e-3, axis=-1,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(FixedBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     name='a',
                                     trainable=False)
        # self.beta = self.add_weight(shape,
        #                             initializer=self.beta_init,
        #                             regularizer=self.beta_regularizer,
        #                             trainable=False)
        # self.running_mean = self.add_weight(shape, initializer='zero',
        #                                     trainable=False)
        # self.running_std = self.add_weight(shape, initializer='one',
        #
        #                                    trainable=False)
        print(self.gamma)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):

        assert self.built, 'Layer must be built before being called'
        input_shape = x.shape

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        #
        # x_normed =

        if sorted(reduction_axes) == range(x.dim())[:-1]:
            x_normed = K.batch_normalization(
                x, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # need broadcasting
            broadcast_running_mean = torch.reshape(self.running_mean, broadcast_shape)
            broadcast_running_std = torch.reshape(self.running_std, broadcast_shape)
            broadcast_beta = torch.reshape(self.beta, broadcast_shape)
            broadcast_gamma = torch.reshape(self.gamma, broadcast_shape)
            x_normed = K.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon)

        return x_normed

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(FixedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))