import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import models
from keras.layers import Layer
from datetime import datetime


# Model input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
#
#     V: num_of_vertices
#     T: num_of_timesteps
#     F: num_of_features
#
# Model output: (*, 5)


def F_norm(weight_matrix, Falpha):
    '''
    compute F Norm
    '''
    return Falpha * K.sum(weight_matrix ** 2)


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


class functional_graph_learning(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''

    def __init__(self, alpha, **kwargs):

        self.alpha = alpha
        self.S = tf.convert_to_tensor(0.0)
        self.diff = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(functional_graph_learning, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape

        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        self.add_loss(F_norm_loss(self.S, self.alpha))
        self.add_loss(diff_loss(self.diff, self.S))
        super(functional_graph_learning, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape
        N = tf.shape(x)[0]

        # shape: (N,V,F) use the current slice
        x = x[:, int(int(x.shape[1]) / 2), :, :]

        # shape: (N,V,V)
        diff = tf.transpose(tf.broadcast_to(x, [V, N, V, F]), perm=[2, 1, 0, 3]) - x

        # shape: (N,V,V)
        tmpS = K.exp(K.reshape(K.dot(tf.transpose(K.abs(diff), perm=[1, 0, 2, 3]), self.a), [N, V, V]))

        # shape: (N,V,V)
        S = tmpS / tf.transpose(tf.broadcast_to(K.sum(tmpS, axis=1), [V, N, V]), perm=[1, 2, 0])

        self.S = S
        self.diff = diff
        return S

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices, num_of_vertices)
        return (input_shape[0], input_shape[2], input_shape[2])


class temporal_graph_learning(Layer):
    '''
        Graph structure learning
        --------
        Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
        Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, **kwargs):
        self.S = tf.convert_to_tensor(0.0)
        super(temporal_graph_learning, self).__init__(**kwargs)

    def build(self, input_shape):

        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape

        self.kernal = self.add_weight(name='kernel',
                                      shape=(num_of_features, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)

        self.add_loss(F_norm(self.S, 0.0005))
        super(temporal_graph_learning, self).build(input_shape)

    def call(self, x):
        S1 = K.dot(x,self.kernal)
        S = K.mean(S1,axis=1)
        self.S = S
        return  S

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices, num_of_vertices)
        return (input_shape[0], input_shape[2], input_shape[2])

class graph_conv(Layer):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''

    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(graph_conv, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        x_shape,  S_shape = input_shape
        _, num_of_timesteps, num_of_vertices, num_of_features = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, num_of_features, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)



    def call(self, x):
        x,  W = x
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        I = tf.ones(shape=(tf.shape(x)[0], num_of_vertices, num_of_vertices))

        #Graph Convolution

        D = tf.linalg.diag(K.sum(W, axis=1))
        L = D - W
        lambda_max = 2.0
        L_t = (2 * L) / lambda_max - [tf.eye(int(num_of_vertices))]
        cheb_polynomials = [tf.eye(int(num_of_vertices)), L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            # shape of x is (batch_size, V, F')
            output = tf.zeros(shape=(tf.shape(x)[0], num_of_vertices, self.num_of_filters))

            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[kk]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k = T_k * I

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k, perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output, 1))
        rs = K.relu(K.concatenate(outputs, axis=1))
        return rs

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_of_filters)
        # return (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][2])

class graph_conv_with_jk(Layer):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_filters),
            S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''

    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(graph_conv_with_jk, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        gcn_shape, S_shape = input_shape
        _, num_of_timesteps, num_of_vertices, num_of_filters = gcn_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=( self.k, self.num_of_filters ,self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(graph_conv_with_jk, self).build(input_shape)

    def call(self, x):

        gcn, W = x

        _, num_of_timesteps, num_of_vertices, num_of_filters = gcn.shape
        I = tf.ones(shape=(tf.shape(gcn)[0],num_of_vertices,num_of_vertices))

        outputs = []

        D = tf.linalg.diag(K.sum(W, axis=1))
        L = D - W
        lambda_max = 2.0
        L_t = (2 * L) / lambda_max - [tf.eye(int(num_of_vertices))]
        cheb_polynomials = [tf.eye(int(num_of_vertices)), L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

        for time_step in range(num_of_timesteps):
            # shape of x is (batch_size, V, F)
            graph_signal = gcn[:, time_step, :, :]
            # shape of x is (batch_size, V, F')

            output = tf.zeros(shape=(tf.shape(gcn)[0], num_of_vertices, self.num_of_filters))
            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[kk]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k = T_k * I

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k, perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output, 1))
        gcn2 = K.relu(K.concatenate(outputs, axis=1))
        #jump knowledge
        jkgcn = gcn2  + K.sigmoid(gcn)
        return jkgcn

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_of_filters)


class LayerNorm(tf.keras.layers.Layer):
    # rewritten to work with newer tensorflow version
    def __init__(self, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=3)

    def call(self, x):
        x_residual, time_conv_output = x
        relu_x = K.relu(x_residual + time_conv_output)
        ln = self.layer_norm(relu_x)
        return ln


def JKSTGCNBlock(x, k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                    time_conv_kernel,  i=0):

    #functional graph learning
    A_1 = functional_graph_learning(alpha=0.0005)(x)
    A_1 = layers.Dropout(0.3)(A_1)

    # first graph convolution layer with function based adaptive graph
    gcn = graph_conv(num_of_filters=num_of_chev_filters, k=k)([x, A_1])

    # temporal graph learning
    A_2 = temporal_graph_learning()(x)
    A_2 = layers.Dropout(0.5)(A_2)

    # jumping knowledge graph convolution with temporal information based adaptive graph
    jkgcn = graph_conv_with_jk(num_of_filters=num_of_chev_filters, k=k)([gcn, A_2])

    block_out = layers.Dropout(0.5)(jkgcn)

    time_conv_output = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(time_conv_kernel, 1),
        padding='same',
        strides=(time_conv_strides, 1)
    )(block_out)

    x_residual = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(1, 1),
        strides=(1, time_conv_strides)
    )(block_out)

    # changed to work with newer tensorflow version
    end_output = LayerNorm(name='layer_norm' + str(i))([x_residual, time_conv_output])
    return end_output


def build_JKSTGCN(k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                        time_conv_kernel,
                        sample_shape,  dense_size, opt=None,  regularizer=None, dropout=0, dataset='ISRUC', idx_run=0, idx_fold=0):
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = layers.Input(shape=sample_shape)

    block_out = JKSTGCNBlock(data_layer, k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                             time_conv_kernel)

    block_out = layers.Flatten()(block_out)

    for size in dense_size:
        block_out = layers.Dense(size,activation='relu',kernel_regularizer=regularizer)(block_out)

    # dropout
    if dropout != 0:
        block_out = layers.Dropout(dropout)(block_out)

    softmax = layers.Dense(5, activation='softmax')(block_out)

    model = models.Model(inputs=data_layer, outputs=softmax)

    if opt is None:
        # no optimizer specified for evaluation, use dummy optimizer for model compilation
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['acc'],
        )
    else:
        model._name = f'JKSTGCN_{dataset}_run_{idx_run}_fold_{idx_fold}_time_{datetime.now().strftime("%Y-%m-%d__%H-%M-%S")}'
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc'],
        )
    return model
