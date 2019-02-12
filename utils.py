import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ********** WARMUP **********
def sample_data():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, 100))


def density_model(n_classes=100):
    # define placeholders
    features_ph = tf.placeholder(dtype=tf.int64, shape=(None,))
    one_hot_features = tf.one_hot(features_ph, n_classes)

    # model
    thetas = tf.get_variable("thetas", shape=(n_classes,), dtype=tf.float32)
    probs = tf.exp(thetas) / tf.reduce_sum(tf.exp(thetas))
    log2_probs = tf.log(probs) / np.log(2)

    # loss
    avg_neg_log_lik = -tf.reduce_mean(one_hot_features * log2_probs) * n_classes

    # minimizer
    train_op = tf.train.GradientDescentOptimizer(0.02).minimize(avg_neg_log_lik)

    return features_ph, probs, train_op, avg_neg_log_lik


def get_train_val_test(data):
    """
    Cuts data into train/val/test (64%/12%/20%)
    """
    t_data, test_data = data[0:int(len(data)*80/100)], data[int(len(data)*80/100):]
    indices = np.arange(0, len(t_data))
    np.random.shuffle(indices)
    train_data, val_data = t_data[indices[:(len(t_data)*80)//100]],\
                           t_data[indices[(len(t_data)*80)//100:]]
    return train_data, val_data, test_data


def data_batch_iterator(data, batch_size):
    while True:
        indices = np.arange(0, len(data))
        np.random.shuffle(indices)
        for index in range(0, len(indices), batch_size):
            yield data[indices[index:index + batch_size]]


def _F_inverse(cumsum_probs, u):
    for index, value in enumerate(cumsum_probs):
        if value > u:
            return index


def generate_data(probs, n_examples=1000):
    cumsum = np.cumsum(probs)
    inverse = np.vectorize(lambda u: _F_inverse(cumsum, u))
    random_values = np.random.rand(n_examples)
    return inverse(random_values)


def plot_loss(steps, train_losses, val_losses):
    plt.figure(figsize=(15, 5))
    plt.plot(steps, train_losses, label="training set")
    plt.plot(steps, val_losses, label="validation set")

    plt.title("Average Negative Log Likelihood evolution in training")
    plt.xlabel("steps")
    plt.ylabel("average negative log likelihood")
    plt.legend()


# ********** TWO-DIMENSIONAL DATA **********
# Conditional model
def _1d_to_2d(value, n_cols):
    return value % n_cols, value // n_cols


def generate_from_2d(probs, n_examples=100000):
    n_cols = np.shape(probs)[1]
    flat_probs = probs.reshape(-1)

    to_2d = np.vectorize(lambda value: _1d_to_2d(value, n_cols))

    values_1d = np.random.choice(np.arange(0, 200 * 200), size=n_examples, p=flat_probs)
    values_2d = to_2d(values_1d)

    return np.transpose(values_2d)


def conditional_model(one_hot=True, n_layers=2, activation=tf.nn.swish):
    # Define kernel initializer
    initializer = tf.initializers.random_uniform(-0.01, 0.01)

    # Define placeholder
    features_ph = tf.placeholder(dtype=tf.int64, shape=(None,))
    labels_ph = tf.placeholder(dtype=tf.int64)
    dropout_ph = tf.placeholder(dtype=tf.float32)

    # model
    if one_hot:
        layer = tf.one_hot(features_ph, depth=200)
    else:
        layer = tf.cast(tf.expand_dims(features_ph, axis=-1), tf.float32)

    for _ in range(n_layers):
        layer = tf.layers.dense(layer, 200, activation=activation,
                                use_bias=True, kernel_initializer=initializer)
        layer = tf.layers.dropout(layer, dropout_ph)
    logits = tf.layers.dense(layer, 200, activation=None,
                             use_bias=False, kernel_initializer=initializer)
    probs = tf.nn.softmax(logits)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels_ph, logits)

    # train op
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return features_ph, labels_ph, dropout_ph, probs, loss, train_op


# MADE
def made_model(one_hot=True, n_layers=2, activation=tf.nn.swish):
    # Define random uniform initializer
    initializer = tf.initializers.random_uniform(-0.01, 0.01)

    # define placeholder
    features_ph = tf.placeholder(dtype=tf.int64, shape=(None, 2))
    labels_ph = tf.placeholder(dtype=tf.int64, shape=(None, 2))
    dropout_ph = tf.placeholder(dtype=tf.float32, shape=())
    batch_size = tf.shape(features_ph)[0]

    # The first input is ignored (never fed in any neuron), so we can
    # only consider the second one
    input_feature = features_ph[:, 0]

    if one_hot:
        layer = tf.one_hot(input_feature, depth=200)
    else:
        layer = tf.cast(tf.expand_dims(input_feature, axis=-1), tf.float32)

    for _ in range(n_layers):
        layer = tf.layers.dense(layer, 200, activation=activation,
                                use_bias=True, kernel_initializer=initializer)
        layer = tf.layers.dropout(layer, dropout_ph)
    logits = tf.layers.dense(layer, 200, activation=None,
                             use_bias=False, kernel_initializer=initializer)

    # The last neuron is not fed by any previous neurons
    lonely_neuron = tf.get_variable("lonely_neuron", shape=(200,), dtype=tf.float32,
                                    initializer=initializer)
    tiled_lonely_neuron = tf.reshape(tf.tile(lonely_neuron, [batch_size]), (batch_size, 200))

    last_layer = tf.stack([tiled_lonely_neuron, logits], axis=1)
    probs = tf.nn.softmax(last_layer)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels_ph, last_layer)

    # train op
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return features_ph, labels_ph, dropout_ph, probs, loss, train_op


# ********** HIGH-DIMENSIONAL DATA **********

def get_4d_mask(mask_shape, mask_type):
    """
    Get the 4d convolution mask of type mask_type and shape mask_shape.
    :mask_shape: height, width, n_inputs, n_outputs
    """
    assert (len(mask_shape) == 4)
    h, w, in_, out = mask_shape
    # assert(in_ % 3 == 0)
    # assert(out % 3 == 0)

    if mask_type == "A":
        base_block = np.array([[0.0, 1.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0]])
    elif mask_type == "B":
        base_block = np.array([[1.0, 1.0, 1.0],
                               [0.0, 1.0, 1.0],
                               [0.0, 0.0, 1.0]])
    else:
        raise ValueError("mask type should be A or B")

    mask = np.ones((h, w, in_, out))
    mask[h // 2 + 1:h, :, :, :] = 0  # zeros on second half of rows
    mask[h // 2, w // 2:w, :, :] = 0  # zeros on second half of columns on middle row
    mask[h // 2, w // 2, :, :] = np.vstack(
        [np.tile(base_block, (out + 2) // 3)[:, :out] for _ in range((in_ + 2) // 3)]
    )[:in_, :]

    return mask


def residual_block(input_, hidden_size):
    """
    input_ --> 1x1 conv B Relu --> 3x3 conv B Relu --> 1x1 conv B Relu --> output
    """
    input_size = (input_).shape[-1]

    with tf.variable_scope("residual_block", reuse=tf.AUTO_REUSE):
        # 1x1 conv B Relu
        weights_1 = tf.get_variable("conv1_weights", shape=(1, 1, input_size, hidden_size // 2), dtype=tf.float32)
        masked_weights_1 = weights_1 * get_4d_mask((1, 1, input_size, hidden_size // 2), "B")
        conv1 = tf.nn.conv2d(input_, masked_weights_1, strides=[1, 1, 1, 1], padding="SAME")
        out1 = tf.nn.relu(conv1)

        # 3x3 conv B Relu
        weights_2 = tf.get_variable("conv2_weights", shape=(3, 3, hidden_size // 2, hidden_size // 2), dtype=tf.float32)
        masked_weights_2 = weights_2 * get_4d_mask((3, 3, hidden_size // 2, hidden_size // 2), "B")
        conv2 = tf.nn.conv2d(out1, masked_weights_2, strides=[1, 1, 1, 1], padding="SAME")
        out2 = tf.nn.relu(conv2)

        # 1x1 conv B Relu
        weights_3 = tf.get_variable("conv3_weights", shape=(1, 1, hidden_size // 2, hidden_size), dtype=tf.float32)
        masked_weights_3 = weights_3 * get_4d_mask((1, 1, hidden_size // 2, hidden_size), "B")
        conv3 = tf.nn.conv2d(out2, masked_weights_3, strides=[1, 1, 1, 1], padding="SAME")
        out3 = tf.nn.relu(conv3)

    return out3 + input_


def pixelcnn(hidden_size=128):
    # define placeholders
    features_ph = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 3))
    labels_ph = tf.placeholder(dtype=tf.int64, shape=(None, 28, 28, 3))

    # model
    # 7x7 conv A Relu
    weights_A = tf.get_variable("convA_weights", shape=(1, 1, 3, hidden_size), dtype=tf.float32)
    masked_weights_A = weights_A * get_4d_mask((1, 1, 3, hidden_size), "A")
    convA = tf.nn.conv2d(features_ph, masked_weights_A, strides=[1, 1, 1, 1], padding="SAME")
    layer = tf.nn.tf.nn.relu(convA)
    layer = tf.contrib.layers.layer_norm(layer)

    # 12 Residual Blocks
    for _ in range(12):
        layer = residual_block(layer, hidden_size)
        layer = tf.contrib.layers.layer_norm(layer)

    # 1x1 conv B Relu
    weights_B1 = tf.get_variable("convB1_weights", shape=(1, 1, hidden_size, hidden_size), dtype=tf.float32)
    masked_weights_B1 = weights_B1 * get_4d_mask((1, 1, hidden_size, hidden_size), "B")
    convB1 = tf.nn.conv2d(layer, masked_weights_B1, strides=[1, 1, 1, 1], padding="SAME")
    outB1 = tf.nn.relu(convB1)
    outB1 = tf.contrib.layers.layer_norm(outB1)

    # 1x1 conv B Relu
    weights_B2 = tf.get_variable("convB2_weights", shape=(1, 1, hidden_size, 3 * hidden_size), dtype=tf.float32)
    masked_weights_B2 = weights_B2 * get_4d_mask((1, 1, hidden_size, 3 * hidden_size), "B")
    logits = tf.nn.conv2d(outB1, masked_weights_B2, strides=[1, 1, 1, 1], padding="SAME")

    logits = tf.reshape(logits, [-1, 28, 28, hidden_size, 3])
    logits = tf.transpose(logits, [0, 1, 2, 4, 3])

    # softmax layer
    probs = tf.nn.softmax(logits)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels_ph, logits)

    # train op
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return features_ph, labels_ph, probs, loss, train_op


def pixelcnn_made(hidden_size=128):
    # define placeholders
    features_ph = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 3))
    labels_ph = tf.placeholder(dtype=tf.int64, shape=(None, 28, 28, 3))

    # model
    # 7x7 conv A Relu
    weights_A = tf.get_variable("convA_weights", shape=(1, 1, 3, hidden_size), dtype=tf.float32)
    masked_weights_A = weights_A * get_4d_mask((1, 1, 3, hidden_size), "A")
    convA = tf.nn.conv2d(features_ph, masked_weights_A, strides=[1, 1, 1, 1], padding="SAME")
    layer = tf.nn.relu(convA)
    layer = tf.contrib.layers.layer_norm(layer)

    # 12 Residual Blocks
    for _ in range(12):
        layer = residual_block(layer, hidden_size)
        layer = tf.contrib.layers.layer_norm(layer)

    # 1x1 conv B Relu
    weights_B1 = tf.get_variable("convB1_weights", shape=(1, 1, hidden_size, hidden_size), dtype=tf.float32)
    masked_weights_B1 = weights_B1 * get_4d_mask((1, 1, hidden_size, hidden_size), "B")
    convB1 = tf.nn.conv2d(layer, masked_weights_B1, strides=[1, 1, 1, 1], padding="SAME")
    outB1 = tf.nn.relu(convB1)
    outB1 = tf.contrib.layers.layer_norm(outB1)

    # 1x1 conv B Relu
    weights_B2 = tf.get_variable("convB2_weights", shape=(1, 1, hidden_size, hidden_size), dtype=tf.float32)
    masked_weights_B2 = weights_B2 * get_4d_mask((1, 1, hidden_size, hidden_size), "B")
    pixel_output = tf.nn.conv2d(outB1, masked_weights_B2, strides=[1, 1, 1, 1], padding="SAME")

    # MADE
    red_logits, green_logits, blue_logits = made_block(features_ph, pixel_output, hidden_size=128)
    logits = tf.concat([red_logits, green_logits, blue_logits], axis=-1)
    logits = tf.reshape(logits, (-1, 28, 28, 3, 4))

    # softmax layer
    probs = tf.nn.softmax(logits)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels_ph, logits)

    # train op
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return features_ph, labels_ph, probs, loss, train_op


def rescale_img(img):
    return ((img - np.min(img))/(np.max(img) - np.min(img)) * 255).astype(int)


def generate_pixel(img, probs, i, j):
    pixel_probs = probs[i, j]
    pixel_value_R = np.random.choice(np.arange(0, 4), p=pixel_probs[0])
    pixel_value_G = np.random.choice(np.arange(0, 4), p=pixel_probs[1])
    pixel_value_B = np.random.choice(np.arange(0, 4), p=pixel_probs[2])
    img[i, j] = [pixel_value_R, pixel_value_G, pixel_value_B]
    return img


