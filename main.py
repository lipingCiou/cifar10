import tensorflow_datasets as tfds
import numpy as np
import jax,gc
from jax import numpy as jnp
from jax import random
from jax import nn
from functools import partial


# 加載 Cifar10 數據集
def load_cifar10():
    ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, batch_size=-1)
    train_data = tfds.as_numpy(ds_train)
    test_data = tfds.as_numpy(ds_test)
    x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]

    # 正規化到 [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test


# 初始化 CNN 參數
def initialize_params(rng_key):
    key1, key2, key3, key4, key5 = random.split(rng_key, 5)

    params = {
        # 卷積層1：輸入通道=3, 濾波器=32, 核大小=3x3
        'W1': random.normal(key1, (3, 3, 3, 32)) * jnp.sqrt(1 / 3),

        'b1': jnp.zeros(32),

        # 卷積層2：輸入通道=32, 濾波器=64, 核大小=3x3
        'W2': random.normal(key2, (3, 3, 32, 64)) * jnp.sqrt(2 / 32),
        'b2': jnp.zeros(64),
        'W3': random.normal(key3, (3, 3, 64, 128)) * jnp.sqrt(1 / 64),  # 新增 128 通道的卷積層
        'b3': jnp.zeros(128),
        # 全連接層1：輸入 4x4x128 (經過池化後), 輸出 256
        'W4': random.normal(key3, (4 * 4 * 128, 256)) * jnp.sqrt(2 / (4 * 4 * 128)),
        'b4': jnp.zeros(256),

        # 全連接層2：輸入 128, 輸出 10 類別
        'W5': random.normal(key4, (256, 10)) * jnp.sqrt(2 / 128),
        'b5': jnp.zeros(10)
    }

    return params

# bn
def batch_norm(x, axis=(0, 1, 2), eps=1e-5):
    """對輸入 x 進行 Batch Normalization"""
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)

# CNN 前向傳播
def cnn_forward(x, params, train=True):
    # 第一層卷積 (3x3, 32個濾波器) + ReLU + BN
    x = jax.lax.conv_general_dilated(x, params['W1'], window_strides=(1, 1),
                                      padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b1']
    x = nn.relu(x)
    x = batch_norm(x)  # 加入 Batch Normalization

    # 池化層1 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # 第二層卷積 (3x3, 64個濾波器) + ReLU + BN
    x = jax.lax.conv_general_dilated(x, params['W2'], window_strides=(1, 1),
                                      padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b2']
    x = nn.relu(x)
    x = batch_norm(x)  # 加入 Batch Normalization

    # 池化層2 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # 第三層卷積 (3x3, 128個濾波器) + ReLU + BN
    x = jax.lax.conv_general_dilated(x, params['W3'], window_strides=(1, 1),
                                     padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b3']
    x = nn.relu(x)
    x = batch_norm(x)  # 加入 Batch Normalization

    # 池化層2 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # 展平成 1D
    x = x.reshape(x.shape[0], -1)  # 這裡展平後應該是 (batch_size, 2048)

    # 全連接層1 (128) + ReLU + BN
    x = jnp.dot(x, params['W4']) + params['b4']
    x = nn.relu(x)
    x = batch_norm(x, axis=0)  # 加入 Batch Normalization

    # Dropout（僅在訓練時啟用）
    dropout_rate = 0.3
    if train:
        x = x * (random.bernoulli(random.PRNGKey(42), p=1-dropout_rate, shape=x.shape) / (1 - dropout_rate))

    # 輸出層 (10 類別)
    x = jnp.dot(x, params['W5']) + params['b5']

    return x



# 交叉熵損失
def loss(params, x, y):
    y_pred = cnn_forward(x, params)
    log_probs = jnp.log(nn.softmax(y_pred) + 1e-7)
    return -jnp.mean(jnp.sum(y * log_probs, axis=1))


# One-hot 編碼
def one_hot(labels, num_classes=10):
    return jnp.eye(num_classes)[labels]


# 批次生成
def get_batches(data, labels, batch_size):
    num_samples = len(data)
    indices = np.random.permutation(num_samples)
    data = data[indices]
    labels = labels[indices]
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        yield data[start:end], labels[start:end]


# 更新參數
def update_params(params, grad, learning_rate):
    return {k: params[k] - learning_rate * grad[k] for k in params}


# 計算準確率
def accuracy(params, x_test, y_test):
    y_pred = cnn_forward(x_test, params)
    predicted_classes = jnp.argmax(y_pred, axis=1)
    true_classes = jnp.argmax(y_test, axis=1)  # 由於使用 One-Hot，需取 argmax
    return jnp.mean(predicted_classes == true_classes)


# 主訓練腳本
if __name__ == '__main__':
    rng_key = random.PRNGKey(42)
    numEpochs = 20
    batch_size = 64
    initial_lr = 0.01
    decay_rate = 0.95

    num_classes = 10

    x_train, y_train, x_test, y_test = load_cifar10()
    y_train, y_test = one_hot(y_train), one_hot(y_test)

    params = initialize_params(rng_key)
    compute_loss_and_grad = partial(jax.value_and_grad(loss))

    for epoch in range(numEpochs):
        for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
            loss_value, grad = compute_loss_and_grad(params, x_batch, y_batch)
            learning_rate = initial_lr * (decay_rate ** epoch)
            params = update_params(params, grad, learning_rate)
        train_acc = accuracy(params, x_train[:1000], y_train[:1000])
        test_acc = accuracy(params, x_test[:1000], y_test[:1000])
        print(f'週期 {epoch + 1}, 損失: {loss_value:.4f}, 訓練準確率: {train_acc:.4f}, 測試準確率: {test_acc:.4f}')
        jax.clear_caches()
        gc.collect()

    final_test_acc = accuracy(params, x_test, y_test)
    print(f'最終測試準確率: {final_test_acc:.4f}')
