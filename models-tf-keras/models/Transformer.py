import numpy as np
import tensorflow as tf
from tensorflow import keras


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.projection_dim = embed_dim // num_heads

        self.query_dense = keras.layers.Dense(embed_dim, kernel_regularizer=keras.regularizers.l1_l2(0.01))
        self.key_dense = keras.layers.Dense(embed_dim, kernel_regularizer=keras.regularizers.l1_l2(0.01))
        self.value_dense = keras.layers.Dense(embed_dim, kernel_regularizer=keras.regularizers.l1_l2(0.01))

        self.combine_heads = keras.layers.Dense(embed_dim, kernel_regularizer=keras.regularizers.l1_l2(0.01))

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)
        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape == (batch_size , seq_len, embed_dim)

        batch_size = tf.shape(inputs)[0]

        # (batch_size , seq_len, embed_dim)
        query = self.query_dense(inputs)

        # (batch_size , seq_len_qk_v , embed_dim)
        # (seq_len_qk_v == seq_len for Encoder or seq_len_qk_v == 1 for Decoder)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size , num_heads, seq_len, projection_dim)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention_output = self.attention(query, key, value)

        # (batch_size , seq_len_qk_v , embed_dim)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size , seq_len_qk_v , embed_dim)
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))

        # (batch_size , seq_len_qk_v , embed_dim)
        output = self.combine_heads(concat_attention)

        return output


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.attention_layer = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)

        self.attention_norm_layer = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_layer_1 = keras.layers.Dense(ff_dim, activation="relu",
                                             kernel_regularizer=keras.regularizers.l1_l2(0.01))

        self.ff_layer_2 = keras.layers.Dense(embed_dim, kernel_regularizer=keras.regularizers.l1_l2(0.01))

    def call(self, x, y):
        # (batch_num , seq_len , emdbedding_dimension)
        attn_output = self.attention_layer(x, y)  # (batch_num , seq_len_qk_v(if Encoder), embedding_dimension)

        # (batch_num , seq_len , emdbedding_dimension)
        attn_output = self.attention_norm_layer(x + attn_output)

        # (batch_num , seq_len , ff_dim)
        ffn_output = self.ff_layer_1(attn_output)

        # (batch_num , seq_len , emdbedding_dimension)
        ffn_output = self.ff_layer_2(ffn_output)

        # (batch_num , seq_len , emdbedding_dimension)
        # residual connection
        transformer_block_output = self.attention_norm_layer(attn_output + ffn_output)

        return transformer_block_output


def getPositionEncoding(seq_len, dim, n=10000):
    PE = np.zeros(shape=(seq_len, dim))
    for pos in range(seq_len):
        for i in range(int(dim / 2)):
            denominator = np.power(n, 2 * i / dim)
            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)

    return PE


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embedding_dim)
        self.positions = tf.range(start=0, limit=maxlen, delta=1)

    def call(self, x):
        # maxlen = tf.shape(x)[-1]

        # positions = tf.range(start=0, limit=maxlen, delta=1)

        x = self.token_emb(x)

        # combine token embedding and position embedding
        return x + self.pos_emb(self.positions)


def create_transformer(vocab_size, maxlen, num_class, d_model, num_heads, num_layers, ff_dim):
    input_1 = keras.Input(shape=(10, 500))
    input_2 = keras.Input(shape=(100, 100, 10))

    split_inputs = [input_1[:, i, :] for i in range(10)]
    output_block = []

    tb_eeg = TransformerBlock(d_model, num_heads, ff_dim)
    tb_eog = TransformerBlock(d_model, num_heads, ff_dim)
    tb = TransformerBlock(d_model, num_heads, ff_dim)
    PE = TokenAndPositionEmbedding(maxlen, vocab_size=vocab_size, embedding_dim=d_model)
    PE_EEG = TokenAndPositionEmbedding(maxlen, vocab_size=vocab_size, embedding_dim=d_model)
    PE_EOG = TokenAndPositionEmbedding(maxlen, vocab_size=vocab_size, embedding_dim=d_model)
    for i in range(6):
        x = PE_EEG(split_inputs[i])
        for j in range(num_layers):
            x = tb_eeg(x)
        output_block.append(x)

    for i in range(6, 8):
        x = PE_EOG(split_inputs[i])
        for j in range(num_layers):
            x = tb_eog(x)
        output_block.append(x)
    for i in range(8, 10):
        x = PE(split_inputs[i])
        for j in range(num_layers):
            x = tb(x)

        output_block.append(x)

    output_block = tf.concat([output_block[i] for i in range(10)], axis=-1)

    # # conv2_x
    # x = ConvBlock(input_tensor=input_2, num_output=(16, 16), stride=(1, 1), stage_name='1', block_name='a')
    # x = IdentityBlock(input_tensor=x, num_output=(16, 16), stage_name='1', block_name='b')
    # x = SEblock(x)
    # x = ConvBlock(input_tensor=x, num_output=(24, 24), stride=(1, 1), stage_name='2', block_name='c')
    # x = IdentityBlock(input_tensor=x, num_output=(24, 24), stage_name='2', block_name='d')
    # x = SEblock(x)
    # x = ConvBlock(input_tensor=x, num_output=(40, 40), stride=(1, 1), stage_name='3', block_name='e')
    # x = IdentityBlock(input_tensor=x, num_output=(40, 40), stage_name='3', block_name='f')
    # x = SEblock(x)
    x = unet(input_2)
    x = keras.layers.Flatten()(x)
    output_block = keras.layers.GlobalAvgPool1D()(output_block)
    output_block = keras.layers.concatenate([x, output_block])
    output_block = keras.layers.Dropout(0.1)(output_block)

    output_tensor = keras.layers.Dense(num_class, activation='softmax')(output_block)

    return keras.Model(inputs=[input_1, input_2], outputs=output_tensor)


class SMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(SMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output

    def cosine_similarity(self, q, k):
        norm_q = tf.linalg.norm(q, axis=-1)
        norm_k = tf.linalg.norm(k, axis=-1)

        similarity = tf.reduce_sum(q * k, axis=-1) / (norm_q * norm_k + 1e-8)  # add epsilon to avoid division by zero

        return similarity

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # shape: (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # shape: (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # shape: (batch_size, seq_len_v, d_model)

        q = tf.reshape(q,
                       (batch_size, -1, self.num_heads, self.depth))  # shape: (batch_size, seq_len_q, num_heads, depth)
        k = tf.reshape(k,
                       (batch_size, -1, self.num_heads, self.depth))  # shape: (batch_size, seq_len_k, num_heads, depth)
        v = tf.reshape(v,
                       (batch_size, -1, self.num_heads, self.depth))  # shape: (batch_size, seq_len_v, num_heads, depth)

        q = tf.transpose(q, perm=[0, 2, 1, 3])  # shape: (batch_size,num_heads ,seq_len_q ,depth)
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # shape: (batch_size,num_heads ,seq_len_k ,depth)
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # shape: (batch_size,num_heads ,seq_len_v ,depth)

        similarity_scores = self.cosine_similarity(q, k)  # shape: (batch_size,num_heads ,seq_len_q ,seq_len_k )
        attention_weights = tf.nn.softmax(similarity_scores,
                                          axis=-1)  # shape: (batch_size,num_heads ,seq_len_q ,seq_len_k )

        output = tf.matmul(attention_weights, v)  # shape: (batch_size,num_heads ,seq_len_q ,depth )

        output = tf.transpose(output, [0, 2, 1, 3])  # shape:( batch_size,q_seq,k_seq,d_model/num_head )

        output = tf.reshape(output, (batch_size, -1, self.d_model))  # shape:( batch_size,q_seq,d_model )

        output = self.dense(output)  # shape:( batch_size,q_seq,d_model )

        return output


def ConvBlock(input_tensor, num_output, stride, stage_name, block_name):
    filter1, filter2 = num_output

    x = keras.layers.SeparableConv2D(filter1, 3, strides=stride, padding='same',
                                     name='res' + stage_name + block_name + '_branch2a',
                                     kernel_regularizer=keras.regularizers.l1_l2(0.01))(input_tensor)
    x = keras.layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2a')(x)
    x = keras.layers.Activation(tf.nn.elu, name='res' + stage_name + block_name + '_branch2a_relu')(x)

    shortcut = keras.layers.SeparableConv2D(filter2, 1, strides=stride, padding='same',
                                            name='res' + stage_name + block_name + '_branch1',
                                            kernel_regularizer=keras.regularizers.l1_l2(0.01))(input_tensor)
    shortcut = keras.layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch1')(shortcut)

    x = keras.layers.add([x, shortcut], name='res' + stage_name + block_name)
    x = keras.layers.Activation(tf.nn.elu, name='res' + stage_name + block_name + '_relu')(x)

    return x


def IdentityBlock(input_tensor, num_output, stage_name, block_name):
    filter1, filter2 = num_output

    x = keras.layers.SeparableConv2D(filter1, 3, strides=(1, 1), padding='same',
                                     name='res' + stage_name + block_name + '_branch2a',
                                     kernel_regularizer=keras.regularizers.l1_l2(0.01))(input_tensor)
    x = keras.layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2a')(x)
    x = keras.layers.Activation(tf.nn.elu, name='res' + stage_name + block_name + '_branch2a_relu')(x)

    # x = keras.layers.SeparableConv2D(filter2, 3, strides=(1, 1), padding='same',
    #                         name='res' + stage_name + block_name + '_branch2b',kernel_regularizer=keras.regularizers.l1_l2(0.01))(x)
    # x = keras.layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2b')(x)
    # x = keras.layers.Activation('relu', name='res' + stage_name + block_name + '_branch2b_relu')(x)

    shortcut = input_tensor

    x = keras.layers.add([x, shortcut], name='res' + stage_name + block_name)
    x = keras.layers.Activation(tf.nn.elu, name='res' + stage_name + block_name + '_relu')(x)

    return x


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = keras.layers.SeparableConv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def unet(inpt):
    conv1 = Conv2d_BN(inpt, 8, (3, 3))
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 16, (3, 3))
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 32, (3, 3))
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 64, (3, 3))
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 128, (3, 3))
    conv5 = keras.layers.Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 64, (3, 3))
    convt1 = downsample(convt1, [13, 13])
    concat1 = keras.layers.concatenate([conv4, convt1], axis=3)
    concat1 = keras.layers.Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 64, (3, 3))

    convt2 = Conv2dT_BN(conv6, 32, (3, 3))
    convt2 = downsample(convt2, [25, 25])
    concat2 = keras.layers.concatenate([conv3, convt2], axis=3)
    concat2 = keras.layers.Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 32, (3, 3))

    convt3 = Conv2dT_BN(conv7, 16, (3, 3))
    concat3 = keras.layers.concatenate([conv2, convt3], axis=3)
    concat3 = keras.layers.Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 16, (3, 3))

    convt4 = Conv2dT_BN(conv8, 8, (3, 3))
    concat4 = keras.layers.concatenate([conv1, convt4], axis=3)
    concat4 = keras.layers.Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 8, (3, 3))

    conv9 = keras.layers.Dropout(0.5)(pool4)
    outpt = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(
        conv9)

    return outpt


def downsample(inputs, shape):
    # 获取输入张量的大小
    size = inputs.get_shape().as_list()[1:3]
    # 定义目标大小为4x4
    target_size = shape
    # 计算缩放比例
    scale = [target_size[0] / size[0], target_size[1] / size[1]]
    # 使用双线性插值法进行下采样
    output = tf.image.resize(inputs, target_size, method=tf.image.ResizeMethod.BILINEAR)

    return output


def unet2(inputs):
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = Dropout(0.25)(pool1)
    # pool1 = BatchNormalization()(pool1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    # pool2 = BatchNormalization()(pool2)

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    # pool3 = BatchNormalization()(pool3)

    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # pool4 = BatchNormalization()(pool4)

    conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(256, (2, 2), strides=(
        2, 2), padding='same')(conv5), conv4], axis=3)
    # up6 = Dropout(0.5)(up6)
    # up6 = BatchNormalization()(up6)
    conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    conv3 = downsample(conv3, [24, 24])
    up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(128, (2, 2), strides=(
        2, 2), padding='same')(conv6), conv3], axis=3)
    # up7 = Dropout(0.5)(up7)
    # up7 = BatchNormalization()(up7)
    conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    conv2 = downsample(conv2, [48, 48])
    up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(conv7), conv2], axis=3)
    # up8 = Dropout(0.5)(up8)
    # up8 = BatchNormalization()(up8)
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    conv1 = downsample(conv1, [96, 96])
    up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(conv8), conv1], axis=3)
    # up9 = Dropout(0.5)(up9)
    # up9 = BatchNormalization()(up9)
    conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # conv9 = Dropout(0.5)(conv9)

    conv10 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return conv10

# vocab_size = 1000
# maxlen = 50
# num_class = 10
# d_model = 64
# num_heads = 2
# num_layers = 2
# ff_dim = 512
#
# model = create_transformer(
#     vocab_size=vocab_size,
#     maxlen=maxlen,
#     num_class=num_class,
#     d_model=d_model,
#     num_heads=num_heads,
#     num_layers=num_layers,
#     ff_dim=ff_dim
# )

