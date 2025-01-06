import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import models
from keras.layers import Layer
from keras.layers.core import Dropout, Lambda

'''
Model code of MSTGCN.
--------
Model input:  (*, T, V, F)
    T: num_of_timesteps
    V: num_of_vertices
    F: num_of_features
Model output: (*, 5)
'''


################################################################################################
################################################################################################
# Attention Layers

class TemporalAttention(Layer):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.U_1 = self.add_weight(name='U_1',
                                   shape=(num_of_vertices, 1),
                                   initializer='uniform',
                                   trainable=True)
        self.U_2 = self.add_weight(name='U_2',
                                   shape=(num_of_features, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        self.U_3 = self.add_weight(name='U_3',
                                   shape=(num_of_features, ),
                                   initializer='uniform',
                                   trainable=True)
        self.b_e = self.add_weight(name='b_e',
                                   shape=(1, num_of_timesteps, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        self.V_e = self.add_weight(name='V_e',
                                   shape=(num_of_timesteps, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        super(TemporalAttention, self).build(input_shape)

    @tf.function
    def call(self, x):
        _, T, V, F = x.shape

        # shape of lhs is (batch_size, V, T)
        lhs = K.dot(tf.transpose(x, perm=[0,1,3,2]), self.U_1)
        lhs = tf.reshape(lhs, [tf.shape(x)[0], T, F])
        lhs = K.dot(lhs, self.U_2)

        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.U_3, tf.transpose(x,perm=[2,0,3,1]))
        rhs = tf.transpose(rhs, perm=[1,0,2])

        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)

        S = tf.transpose(K.dot(self.V_e, tf.transpose(K.sigmoid(product + self.b_e),perm=[1, 2, 0])),perm=[2, 0, 1])

        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])


class SpatialAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                   shape=(num_of_timesteps, 1),
                                   initializer='uniform',
                                   trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                   shape=(num_of_features, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                   shape=(num_of_features, ),
                                   initializer='uniform',
                                   trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                   shape=(1, num_of_vertices, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                   shape=(num_of_vertices, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        super(SpatialAttention, self).build(input_shape)

    @tf.function
    def call(self, x):
        _, T, V, F = x.shape

        # shape of lhs is (batch_size, V, T)
        lhs = K.dot(tf.transpose(x, perm=[0,2,3,1]), self.W_1)
        lhs = tf.reshape(lhs,[tf.shape(x)[0], V, F])
        lhs = K.dot(lhs, self.W_2)

        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.W_3, tf.transpose(x, perm=[1,0,3,2]))
        rhs = tf.transpose(rhs, perm=[1,0,2])

        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)

        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s),perm=[1, 2, 0])),perm=[2, 0, 1])

        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[2])


################################################################################################
################################################################################################
# Adaptive Graph Learning Layer

def diff_loss(diff, S):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return K.mean(K.sum(K.sum(diff**2, axis=3) * S, axis=(1, 2)))
    else:
        return K.sum(K.sum(diff**2, axis=2) * S)


def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return Falpha * K.sum(K.mean(S**2, axis=0))
    else:
        return Falpha * K.sum(S**2)


class Graph_Learn(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(Graph_Learn, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        # add loss L_{graph_learning} in the layer
        self.add_loss(F_norm_loss(self.S, self.alpha))
        self.add_loss(diff_loss(self.diff, self.S))
        super(Graph_Learn, self).build(input_shape)

    @tf.function
    def call(self, x):
        _, T, V, F = x.shape
        N = tf.shape(x)[0]

        outputs = []
        diff_tmp = 0
        for time_step in range(T):
            # shape: (N,V,F) use the current slice
            xt = x[:, time_step, :, :]
            # shape: (N,V,V)
            diff = tf.transpose(tf.broadcast_to(xt, [V,N,V,F]), perm=[2,1,0,3]) - xt
            # shape: (N,V,V)
            tmpS = K.exp(K.reshape(K.dot(tf.transpose(K.abs(diff), perm=[1,0,2,3]), self.a), [N,V,V]))
            # normalization
            S = tmpS / tf.transpose(tf.broadcast_to(K.sum(tmpS, axis=1), [V,N,V]), perm=[1,2,0])

            diff_tmp += K.abs(diff)
            outputs.append(S)

        outputs = tf.transpose(outputs, perm=[1,0,2,3])
        self.S = K.mean(outputs, axis=0)
        self.diff = K.mean(diff_tmp, axis=0) /tf.convert_to_tensor(int(T), tf.float32)
        return outputs

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices,num_of_vertices, num_of_vertices)
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[2])


################################################################################################
################################################################################################
# GCN layers

class cheb_conv_with_Att_GL(Layer):
    '''
    K-order chebyshev graph convolution with attention after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             Att (batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(cheb_conv_with_Att_GL, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape, Att_shape, S_shape = input_shape
        _, T, V, F = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, F, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_Att_GL, self).build(input_shape)

    @tf.function
    def call(self, x):
        #Input:  [x, Att, S]
        assert isinstance(x, list)
        assert len(x)==3, 'Cheb_gcn input error'
        x, Att, S = x
        _, T, V, F = x.shape

        S = K.minimum(S, tf.transpose(S,perm=[0,1,3,2])) # Ensure symmetry

        # GCN
        outputs=[]
        for time_step in range(T):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            # print(tf.shape(x1)[0])
            # output = tf.convert_to_tensor(np.zeros([int(tf.shape(x1)[0]), 10, 64]), dtype=tf.float32)#self.num_of_filters
            output = tf.zeros(shape=(tf.shape(x)[0], V, 64))#self.num_of_filters tf.shape(x1)[0]

            A = S[:, time_step, :, :]
            #Calculating Chebyshev polynomials (let lambda_max=2)
            D = tf.linalg.diag(K.sum(A, axis=1))
            L = D - A
            L_t = L - [tf.eye(int(V))]
            cheb_polynomials = [tf.eye(int(V)), L_t]
            for i in range(2, self.k):
                cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

            for kk in range(self.k):
                T_k = cheb_polynomials[kk]              # shape of T_k is (V, V)
                T_k_with_at = T_k * Att                 # shape of T_k_with_at is (batch_size, V, V)
                theta_k = self.Theta[kk]                # shape of theta_k is (F, num_of_filters)

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at, perm=[0, 2, 1]), graph_signal)
                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output,-1))

        return tf.transpose(K.relu(K.concatenate(outputs, axis=-1)), perm=[0,3,1,2])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_of_filters)


class cheb_conv_with_Att_static(Layer):
    '''
    K-order chebyshev graph convolution with static graph structure
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             Att (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, cheb_polynomials, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = tf.cast(cheb_polynomials, tf.float32)
        super(cheb_conv_with_Att_static, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape, Att_shape = input_shape
        _, T, V, F = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, F, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_Att_static, self).build(input_shape)

    @tf.function
    def call(self, x):
        #Input:  [x, Att]
        assert isinstance(x, list)
        assert len(x) == 2, 'cheb_gcn error'
        x, Att = x
        _, T, V, F = x.shape

        outputs = []
        for time_step in range(T):
            # shape is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            output = tf.zeros(shape=(tf.shape(x)[0], V, self.num_of_filters))

            for kk in range(self.k):
                T_k = self.cheb_polynomials[kk]          # shape of T_k is (V, V)
                T_k_with_at = K.dropout(T_k * Att, 0.6)  # shape of T_k_with_at is (batch_size, V, V)
                theta_k = self.Theta[kk]                 # shape of theta_k is (F, num_of_filters)

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at, perm=[0, 2, 1]), graph_signal)
                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output, -1))

        return tf.transpose(K.relu(K.concatenate(outputs, axis=-1)), perm=[0, 3, 1, 2])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_of_filters)

    def get_config(self):
        config = super(cheb_conv_with_Att_static, self).get_config()
        config.update({
            'num_of_filters': self.num_of_filters,
            'k': self.k,
            'cheb_polynomials': self.cheb_polynomials.numpy().tolist()  # Ensure cheb_polynomials is JSON-serializable
            # Add any other parameters needed to recreate the layer
        })
        return config

################################################################################################
################################################################################################
# Some operations

def reshape_dot(x):
    #Input:  [x,TAtt]
    x, TAtt = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x, perm=[0, 2, 3, 1]),
                       (tf.shape(x)[0], -1, tf.shape(x)[1])), TAtt),
        [-1, x.shape[1], x.shape[2], x.shape[3]]
    )


# def LayerNorm(x):
#     # do the layer normalization
#     relu_x = K.relu(x)
#     # ln = tf.keras.layers.layer_norm(relu_x, begin_norm_axis=3)
#     ln = tf.keras.layers.LayerNormalization(axis=3)(relu_x)
#     return ln


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=3)

    def call(self, inputs):
        relu_x = K.relu(inputs)
        ln = self.layer_norm(relu_x)
        return ln


################################################################################################
################################################################################################
# Gradient Reverse Layer

# def reverse_gradient(X, hp_lambda):
#     """Flips the sign of the incoming gradient during training."""
#     num_calls=1
#     try:
#         reverse_gradient.num_calls =reverse_gradient.num_calls+ 1
#     except AttributeError:
#         reverse_gradient.num_calls = num_calls
#         num_calls=num_calls+1
#
#     grad_name = "GradientReversal_%d" % reverse_gradient.num_calls
#
#     @ops.RegisterGradient(grad_name)
#     def _flip_gradients(op,grad):
#         return [tf.negative(grad) * hp_lambda]
#
#     g = K.get_session().graph
#     with g.gradient_override_map({'Identity': grad_name}):
#         y = tf.identity(X)
#
#     return y


@tf.custom_gradient
def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""

    def grad(dy):
        return -hp_lambda * dy, None  # Reverse the gradient and apply hp_lambda

    return X, grad


class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = True
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def build(self, input_shape):
        self.trainable_weights_alt = []

    @tf.function
    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


################################################################################################
################################################################################################
# MSTGCN Block

def MSTGCN_Block(x, k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 cheb_polynomials, time_conv_kernel, GLalpha, i=0):
    '''
    packaged Spatial-temporal convolution Block
    -------
    x: input data;
    k: k-order cheb GCN
    i: block number
    '''

    # temporal attention
    temporal_Att = TemporalAttention()(x)
    x_TAtt = Lambda(reshape_dot, name='reshape_dot'+str(i))([x, temporal_Att])

    # spatial attention
    spatial_Att = SpatialAttention()(x_TAtt)

    # multi-view GCN
    S = Graph_Learn(alpha=GLalpha)(x)
    S = Dropout(0.3)(S)
    spatial_gcn_GL = cheb_conv_with_Att_GL(num_of_filters=num_of_chev_filters, k=k)([x, spatial_Att, S])
    spatial_gcn_SD = cheb_conv_with_Att_static(num_of_filters=num_of_chev_filters, k=k,
                                               cheb_polynomials=cheb_polynomials)([x, spatial_Att])

    # temporal convolution
    time_conv_output_GL = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(time_conv_kernel, 1),
        padding='same',
        strides=(1, time_conv_strides))(spatial_gcn_GL)

    time_conv_output_SD = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(time_conv_kernel, 1),
        padding='same',
        strides=(1, time_conv_strides))(spatial_gcn_SD)

    # LayerNorm
    # end_output_GL = Lambda(LayerNorm, name='layer_norm' + str(2*i))(time_conv_output_GL)
    # end_output_SD = Lambda(LayerNorm, name='layer_norm' + str(2*i+1))(time_conv_output_SD)
    end_output_GL = LayerNorm(name='layer_norm' + str(2*i))(time_conv_output_GL)
    end_output_SD = LayerNorm(name='layer_norm' + str(2*i+1))(time_conv_output_SD)
    return end_output_GL, end_output_SD


################################################################################################
################################################################################################
# MSTGCN

def build_MSTGCN(k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials,
                 time_conv_kernel, sample_shape, num_block, dense_size, opt, GLalpha,
                 regularizer, dropout, lambda_reversal, num_classes=5, num_domain=9):

    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = layers.Input(shape=sample_shape, name='Input_Layer')

    # MSTGCN_Block
    block_out_GL, block_out_SD = MSTGCN_Block(data_layer, k, num_of_chev_filters, num_of_time_filters,
                                              time_conv_strides, cheb_polynomials, time_conv_kernel, GLalpha)
    for i in range(1, num_block):
        block_out_GL, block_out_SD = MSTGCN_Block(block_out_GL, k, num_of_chev_filters, num_of_time_filters,
                                                  time_conv_strides, cheb_polynomials, time_conv_kernel, GLalpha, i)
    block_out = layers.concatenate([block_out_GL, block_out_SD])
    block_out = layers.Flatten()(block_out)

    # dropout
    if dropout != 0:
        block_out = layers.Dropout(dropout)(block_out)

    # Global dense layer
    for size in dense_size:
        dense_out = layers.Dense(size)(block_out)

    # softmax classification
    softmax = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_regularizer=regularizer,
                           name='Label')(dense_out)

    # GRL & G_d
    flip_layer = GradientReversal(lambda_reversal)
    G_d_in = flip_layer(block_out)
    for size in dense_size:
        G_d_out = layers.Dense(size)(G_d_in)
    G_d_out = layers.Dense(units=num_domain,
                           activation='softmax',
                           name='Domain')(G_d_out)

    # training model (with GRL & G_d)
    model = models.Model(inputs=data_layer, outputs=[softmax, G_d_out])
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    # testing model (without GRL & G_d)
    pre_model = models.Model(inputs=data_layer, outputs=softmax)
    pre_model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    return model, pre_model


def build_MSTGCN_test():
    # an example to test
    cheb_k = 3
    num_of_chev_filters = 10
    num_of_time_filters = 10
    time_conv_strides = 1
    time_conv_kernel = 3
    dense_size = np.array([64, 32])
    cheb_polynomials = [np.random.rand(26, 26), np.random.rand(26, 26), np.random.rand(26, 26)]

    model = build_MSTGCN(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials,
                         time_conv_kernel, sample_shape=(5, 26, 9), num_block=1, dense_size=dense_size,
                         opt='adam', useGL=True, GLalpha=0.0001, regularizer=None, dropout=0.0)
    model.summary()
    model.save('MSTGCN_build_test.h5')
    print("save ok")
    return model
