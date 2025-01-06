import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def cVAN_Loss(y_true, y_pred):
    print(y_pred)
    output1 = tf.keras.layers.Lambda(lambda x: x[:, :5])(y_pred)
    # output2 =  tf.keras.layers.Lambda(lambda x:x[:,5:133])(y_pred)
    # output3 =  tf.keras.layers.Lambda(lambda x:x[:,133:])(y_pred)
    print(output1)
    cross_loss = tf.keras.losses.categorical_crossentropy(y_true, output1)
    # cos_loss =  tf.keras.losses.cosine_similarity(output2,output3)
    return cross_loss


def cos_Loss(y_true, y_pred):
    output1 = tf.keras.layers.Lambda(lambda x: x[:, :128])(y_pred)
    output2 = tf.keras.layers.Lambda(lambda x: x[:, 128:])(y_pred)
    # cos_losss =  tf.keras.losses.cosine_similarity(output1,output2)
    output1 = tf.nn.l2_normalize(output1, axis=1)
    output2 = tf.nn.l2_normalize(output2, axis=1)
    margin = 1
    square_pred = tf.square(output1)
    margin_square = tf.square(tf.maximum(margin - output1, 0))
    loss = tf.reduce_mean(output2 * square_pred + (1 - output2) * margin_square)
    return loss


def stander(data):
    standard_scaler = MinMaxScaler()
    data_2d = data.reshape(-1, data.shape[-1])
    normalized_data_standard = standard_scaler.fit_transform(data_2d)
    normalized_data_standard = normalized_data_standard.reshape(data.shape)
    return normalized_data_standard
