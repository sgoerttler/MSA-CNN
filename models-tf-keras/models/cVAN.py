import tensorflow as tf
from tensorflow import keras

from models.Transformer import TransformerBlock, TokenAndPositionEmbedding

from training.loss_cVAN import cos_Loss


def conv_block(input_tensor, feat_out, stride):
    x = keras.layers.SeparableConv2D(feat_out[0], 3, strides=stride, padding='same',
                                     kernel_regularizer=keras.regularizers.l1_l2(0.01))(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.SeparableConv2D(feat_out[0], 3, strides=(1, 1), padding='same',
                                     kernel_regularizer=keras.regularizers.l1_l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    shortcut = keras.layers.SeparableConv2D(feat_out[0], 1, strides=stride,
                                            kernel_regularizer=keras.regularizers.l1_l2(0.01))(input_tensor)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    return x


def identity_block(input_tensor, feat_out):


    x = keras.layers.SeparableConv2D(feat_out[0], 3, strides=(1, 1), padding='same',
                                     kernel_regularizer=keras.regularizers.l1_l2(0.01))(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    shortcut = input_tensor

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    return x


def downsample(inputs):
    # Get the size of the input tensor
    size = inputs.get_shape().as_list()[1:3]
    # Define target size as 4x4
    target_size = [4, 4]
    # Calculating scaling
    scale = [target_size[0] / size[0], target_size[1] / size[1]]
    # Downsampling using bilinear interpolation
    output = tf.image.resize(inputs, target_size, method=tf.image.ResizeMethod.BILINEAR)

    return output


# ResNet18 model
def residualNetworks(input, channels=10, stride=2):
    m = keras.layers.ZeroPadding2D((3, 3))(input)

    # conv1
    x = keras.layers.SeparableConv2D(64, 7, strides=(stride, stride), padding='same', name='conv1')(input)  # 7×7, 64, stride 2
    x = keras.layers.BatchNormalization(name='bn_conv1')(x)
    x = keras.layers.Activation('relu', name='conv1_relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)  # 3×3 max pool, stride 2

    # conv2_x
    x = conv_block(input_tensor=x, feat_out=(64, 64), stride=(1, 1))
    x = identity_block(input_tensor=x, feat_out=(64, 64))
    x = SCAtnn(x)
    scale_x1 = downsample(x)
    cross_x1 = cross_aliment(x)

    # conv3_x
    x = conv_block(input_tensor=x, feat_out=(128, 128), stride=(2, 2))
    x = identity_block(input_tensor=x, feat_out=(128, 128))
    x = SCAtnn(x)
    scale_x2 = downsample(x)
    cross_x2 = cross_aliment(x)

    # conv4_x
    x = conv_block(input_tensor=x, feat_out=(256, 256), stride=(2, 2))
    x = identity_block(input_tensor=x, feat_out=(256, 256))
    x = SCAtnn(x)
    cross_x3 = cross_aliment(x)
    x_skip3 = downsample(x)

    # conv5_x
    x = conv_block(input_tensor=x, feat_out=(512, 512), stride=(2, 2))
    x = identity_block(input_tensor=x, feat_out=(512, 512))
    x = SCAtnn(x)

    x = tf.concat([scale_x1, scale_x2, x], axis=-1)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64 * channels)(x)

    return x, cross_x1, cross_x2, cross_x3





def cross_aliment(input_tensor):
    _, height, width, channels = input_tensor.shape.as_list()
    assert height == width, "height must equal to width"

    max_columns = tf.reduce_max(input_tensor, axis=1)
    data_t = tf.transpose(max_columns, perm=[0, 2, 1])
    # output = tf.reshape(data_t, [, -1])
    output = tf.keras.layers.Flatten()(max_columns)
    output = keras.layers.Dense(500, activation='relu')(output)
    return output


class CosineSimilarityLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarityLoss, self).__init__(**kwargs)

    def cosine_similarity(self, y_true, y_pred):
        y_true_norm = keras.backend.l2_normalize(y_true, axis=-1)
        y_pred_norm = keras.backend.l2_normalize(y_pred, axis=-1)
        similarity = keras.backend.sum(y_true_norm * y_pred_norm, axis=-1)
        return similarity

    def call(self, x, y):
        y_true = x
        y_pred = y
        loss = 1 - self.cosine_similarity(y_true, y_pred)

        self.add_loss(loss, inputs=[x, y])
        return tf.keras.losses.cosine_similarity(x, y)


def cosineloss(x, y):
    similarity = tf.keras.losses.cosine_similarity(x, y)
    return 1 - (similarity + 1) / 2


# Transformer model

def sin(a, b):
    score = tf.matmul(a, b, transpose_b=True)
    dim_key = tf.cast(tf.shape(b)[-1], tf.float32)
    scaled_score = score / tf.math.sqrt(dim_key)

    weights = tf.nn.softmax(scaled_score, axis=-1)

    output = tf.matmul(weights, b)
    return output


# SCAtnn
def SCAtnn(x):
    context_channel = channelAttention(x)
    context_spatial = spatialAttention(x)
    out = keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, )(x, context_spatial, context_channel)
    return out


def channelAttention(x):
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    if channel_axis == -1:
        batch, height, width, channels = keras.backend.int_shape(x)
        assert channels % 2 == 0
        channel = channels // 2
        input_x = keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                                      kernel_initializer='he_normal')(x)
        input_x = keras.layers.Reshape((width * height, channel))(input_x)

        context_mask = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=False,
                                           kernel_initializer='he_normal')(x)
        context_mask = keras.layers.Reshape((width * height, 1))(context_mask)
        context_mask = keras.layers.Softmax(axis=1)(context_mask)
        context = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True))([input_x, context_mask])
        context = keras.layers.Permute((2, 1))(context)
        context = keras.layers.Reshape((1, 1, channel))(context)
        context = keras.layers.Conv2D(channels, kernel_size=1, strides=1, padding='same')(context)

    mask_ch = keras.layers.Activation('sigmoid')(context)
    return keras.layers.Multiply()([x, mask_ch])


def spatialAttention(x):
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    if channel_axis == -1:
        batch, height, width, channels = keras.backend.int_shape(x)
        assert channels % 2 == 0
        channel = channels // 2
        g_x = keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                                  kernel_initializer='he_normal')(
            x)
        avg_x = keras.layers.GlobalAvgPool2D()(g_x)
        avg_x = keras.layers.Softmax()(avg_x)
        avg_x = keras.layers.Reshape((channel, 1))(avg_x)

        theta_x = keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding='same', use_bias=False,
                                      kernel_initializer='he_normal')(x)
        theta_x = keras.layers.Reshape((height * width, channel))(theta_x)
        context = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([theta_x, avg_x])
        context = keras.layers.Reshape((height * width,))(context)
        mask_sp = keras.layers.Activation('sigmoid')(context)
        mask_sp = keras.layers.Reshape((height, width, 1))(mask_sp)

    return keras.layers.Multiply()([x, mask_sp])


def build_cVAN(vocab_size, maxlen, num_class, d_model, num_heads, num_layers, ff_dim, channels=10, n_length=100,
               optimizer=None, weight_logit=None, weight_sim=None):
    # Transformer input
    input_1 = keras.Input(shape=(channels, 500))
    # Residual network input
    input_2 = keras.Input(shape=(n_length, n_length, channels))
    # Residual network output
    x, x_cross1, x_cross2, x_cross3 = residualNetworks(input_2, channels=channels, stride=n_length // 50)
    # split inputs of transformer (10 channels / 4 channels)
    split_inputs = [input_1[:, i, :] for i in range(channels)]
    # each channel's output
    output_block = []

    # transformer block shared param
    TB = TransformerBlock(d_model, num_heads, ff_dim)

    # position ecode shared param
    TP = TokenAndPositionEmbedding(maxlen, vocab_size=vocab_size, embedding_dim=d_model)

    x_cross1 = TP(x_cross1)
    x_cross2 = TP(x_cross2)
    x_cross3 = TP(x_cross3)

    #  Transformer with cross view attention
    for i in range(channels):
        s = TP(split_inputs[i])
        for j in range(num_layers):
            if j == 0:
                s = TB(s, x_cross1)
            elif j == 1:
                s = TB(s, x_cross2)
            else:
                s = TB(s, x_cross3)
        output_block.append(s)

    # concat all outs from transformer
    output_block = tf.concat([output_block[i] for i in range(channels)], axis=-1)

    # globalAvg
    output_block = keras.layers.GlobalAvgPool1D()(output_block)

    m = sin(x, output_block)

    # common feature learning
    x_head1 = x
    t_head1 = output_block
    shared_Dense = keras.layers.Dense(128)
    x_head1 = shared_Dense(x_head1)
    t_head1 = shared_Dense(t_head1)

    # cal cosine loss
    # CosineSimilarityLoss(trainable=True)(x_head1, t_head1)

    # concat cnn-output and transformer output
    output_block = keras.layers.concatenate([x, output_block])

    output_block = keras.layers.Dropout(0.1)(output_block)

    output_tensor = keras.layers.Dense(num_class, activation='softmax', name="logit_output")(output_block)

    cosineLoss = keras.layers.Concatenate(name="sim_output")([x_head1, t_head1])

    model = keras.Model(inputs=[input_1, input_2], outputs=[output_tensor, cosineLoss])

    model.compile(loss={"logit_output": "categorical_crossentropy", "sim_output": cos_Loss},
                  loss_weights={"logit_output": weight_logit, "sim_output": weight_sim},
                  metrics={'logit_output': "accuracy"}, optimizer=optimizer)

    return model