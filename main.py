import tensorflow_datasets as tfds
import numpy as np
import jax, gc
from jax import numpy as jnp
from jax import random
from jax import nn
from functools import partial
import optax  # 使用 Adam 優化器

#  **1. 加載 CIFAR-10 數據集**
def load_cifar10():
    ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, batch_size=-1)
    train_data = tfds.as_numpy(ds_train)
    test_data = tfds.as_numpy(ds_test)
    x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]

    # 標準化到 [-1, 1]
    x_train = x_train.astype(np.float32) / 127.5 - 1
    x_test = x_test.astype(np.float32) / 127.5 - 1

    return x_train, y_train, x_test, y_test


#  **2. 初始化 CNN 參數**
def initialize_params(rng_key):
    keys = random.split(rng_key, 8)  # 生成 8 個子隨機 key

    params = {
        'W1': random.normal(keys[0], (3, 3, 3, 32)) * jnp.sqrt(2 / 3),
        'b1': jnp.zeros(32),
        'W2': random.normal(keys[1], (3, 3, 32, 64)) * jnp.sqrt(2 / 32),
        'b2': jnp.zeros(64),
        'W3': random.normal(keys[2], (3, 3, 64, 128)) * jnp.sqrt(2 / 64),
        'b3': jnp.zeros(128),
        'W4': random.normal(keys[3], (3, 3, 128, 256)) * jnp.sqrt(2 / 128),
        'b4': jnp.zeros(256),
        'W5': random.normal(keys[4], (3, 3, 256, 512)) * jnp.sqrt(2 / 256),
        'b5': jnp.zeros(512),
        'W6': random.normal(keys[5], (512, 512)) * jnp.sqrt(2 / 512),
        'b6': jnp.zeros(512),
        'W7': random.normal(keys[6], (512, 10)) * jnp.sqrt(2 / 512),
        'b7': jnp.zeros(10)
    }
    return params


#  **3. 改進 BN**
def batch_norm(x, gamma, beta, axis=(0, 1, 2), eps=1e-4):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta


#  **4. L2 正則化**
def l2_regularization(params, l2_lambda=1e-5):
    return l2_lambda * sum(jnp.sum(w ** 2) for k, w in params.items() if 'W' in k)


#  **5. CNN 前向傳播**
def cnn_forward(x, params, train=True, rng_key=None):
    gamma, beta = 1.0, 0.0  # BN 參數

    # ✅ 第一層卷積 (3x3, 32個濾波器) + BN + ReLU
    x = jax.lax.conv_general_dilated(
        x, params['W1'], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b1']
    x = nn.relu(batch_norm(x, gamma, beta))

    # ✅ 池化層1 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # ✅ 第二層卷積 (3x3, 64個濾波器) + BN + ReLU
    x = jax.lax.conv_general_dilated(
        x, params['W2'], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b2']
    x = nn.relu(batch_norm(x, gamma, beta))

    # ✅ 池化層2 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # ✅ 第三層卷積 (3x3, 128個濾波器) + BN + ReLU
    x = jax.lax.conv_general_dilated(
        x, params['W3'], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b3']
    x = nn.relu(batch_norm(x, gamma, beta))

    # ✅ 池化層3 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # ✅ 第四層卷積 (3x3, 256個濾波器) + BN + ReLU
    x = jax.lax.conv_general_dilated(
        x, params['W4'], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b4']
    x = nn.relu(batch_norm(x, gamma, beta))

    # ✅ 池化層4 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # ✅ 第五層卷積 (3x3, 512個濾波器) + BN + ReLU
    x = jax.lax.conv_general_dilated(
        x, params['W5'], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + params['b5']
    x = nn.relu(batch_norm(x, gamma, beta))

    # ✅ 池化層5 (2x2, 步幅2)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    # **展平前 Debug**
    #print("Before flatten:", x.shape)  # 這應該是 (batch_size, 1, 1, 512)

    # ✅ 展平成 1D 向量
    x = x.reshape(x.shape[0], -1)  # 這裡展平後應該是 (batch_size, 512)

    # ✅ 全連接層1 (512) + ReLU + BN
    x = jnp.dot(x, params['W6']) + params['b6']
    x = nn.relu(batch_norm(x, gamma, beta, axis=0))  # BN 只在 batch 維度操作

    # ✅ Dropout（僅在訓練時啟用）
    dropout_rate = 0.4
    if train and rng_key is not None:
        rng_key, subkey = random.split(rng_key)
        keep_prob = 1 - dropout_rate
        mask = random.bernoulli(subkey, keep_prob, shape=x.shape)
        x = x * mask / keep_prob  # 避免影響總輸出值

    # ✅ 輸出層 (10 類別)
    x = jnp.dot(x, params['W7']) + params['b7']

    return x


# 交叉熵損失
def loss(params, x, y, l2_lambda=1e-5):
    """ 計算交叉熵損失 + L2 正則化 """
    y_pred = cnn_forward(x, params)
    log_probs = jnp.log(nn.softmax(y_pred) + 1e-7)
    cross_entropy_loss = -jnp.mean(jnp.sum(y * log_probs, axis=1))

    # 加入 L2 正則化
    l2_loss = l2_regularization(params, l2_lambda)

    return cross_entropy_loss + l2_loss


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
    batch_size = 128
    initial_lr = 0.001
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
