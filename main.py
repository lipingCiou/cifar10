import tensorflow_datasets as tfds
import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from jax import nn
from functools import partial



def load_cifar10():
    ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, batch_size=-1)
    train_data = tfds.as_numpy(ds_train)
    test_data = tfds.as_numpy(ds_test)
    x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    return x_train, y_train, x_test, y_test


# 標準化數據
def normalize_data(x_train, x_test):
    red_values = x_train[:, 0:1024].flatten()
    mean_red = np.mean(red_values)
    std_red = np.std(red_values)
    green_values = x_train[:, 1024:2048].flatten()
    mean_green = np.mean(green_values)
    std_green = np.std(green_values)
    blue_values = x_train[:, 2048:3072].flatten()
    mean_blue = np.mean(blue_values)
    std_blue = np.std(blue_values)
    x_train[:, 0:1024] = (x_train[:, 0:1024] - mean_red) / std_red
    x_train[:, 1024:2048] = (x_train[:, 1024:2048] - mean_green) / std_green
    x_train[:, 2048:3072] = (x_train[:, 2048:3072] - mean_blue) / std_blue
    x_test[:, 0:1024] = (x_test[:, 0:1024] - mean_red) / std_red
    x_test[:, 1024:2048] = (x_test[:, 1024:2048] - mean_green) / std_green
    x_test[:, 2048:3072] = (x_test[:, 2048:3072] - mean_blue) / std_blue
    return x_train, x_test


# 初始化參數
def initialize_params(rng_key, input_size, hidden_size1, hidden_size2, output_size):
    key1, key2, key3 = random.split(rng_key, 3)
    stddev_W1 = np.sqrt(2 / input_size)
    W1 = random.normal(key1, (input_size, hidden_size1)) * stddev_W1
    b1 = jnp.zeros(hidden_size1)
    stddev_W2 = np.sqrt(2 / hidden_size1)
    W2 = random.normal(key2, (hidden_size1, hidden_size2)) * stddev_W2
    b2 = jnp.zeros(hidden_size2)
    stddev_W_output = np.sqrt(2 / hidden_size2)
    W_output = random.normal(key3, (hidden_size2, output_size)) * stddev_W_output
    b_output = jnp.zeros(output_size)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W_output': W_output, 'b_output': b_output}


# MLP 模型
def mlp(x, params):
    W1, b1, W2, b2, W_output, b_output = params['W1'], params['b1'], params['W2'], params['b2'], params['W_output'], \
    params['b_output']
    h1 = jnp.dot(x, W1) + b1
    h1 = nn.relu(h1)
    h2 = jnp.dot(h1, W2) + b2
    h2 = nn.relu(h2)
    output = jnp.dot(h2, W_output) + b_output
    return output


# 損失函數
def loss(params, x, y):
    y_pred = mlp(x, params)
    log_probs = jnp.log(nn.softmax(y_pred))
    return -jnp.mean(jnp.sum(y * log_probs, axis=1))


# one-hot 編碼
def one_hot(labels, num_classes):
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
    y_pred = mlp(x_test, params)
    predicted_classes = jnp.argmax(y_pred, axis=1)
    true_classes = y_test
    return jnp.mean(predicted_classes == true_classes)


# 主訓練腳本
if __name__ == '__main__':
    from jax.lib import xla_bridge

    print("JAX 運行平台:", xla_bridge.get_backend().platform)
    print("可用設備:", jax.devices())

    rng_key = random.PRNGKey(42)
    numEpochs = 10
    batch_size = 128
    learning_rate = 0.01
    input_size = 3072
    hidden_size1 = 1024
    hidden_size2 = 512
    output_size = 10
    num_classes = 10

    x_train, y_train, x_test, y_test = load_cifar10()
    x_train, x_test = normalize_data(x_train, x_test)
    params = initialize_params(rng_key, input_size, hidden_size1, hidden_size2, output_size)
    compute_loss_and_grad = partial(jax.value_and_grad(loss))

    for epoch in range(numEpochs):
        for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
            y_batch_one_hot = one_hot(y_batch, num_classes)
            loss_value, grad = compute_loss_and_grad(params, x_batch, y_batch_one_hot)
            params = update_params(params, grad, learning_rate)
        train_acc = accuracy(params, x_train[:1000], y_train[:1000])
        test_acc = accuracy(params, x_test[:1000], y_test[:1000])
        print(f'週期 {epoch + 1}, 損失: {loss_value}, 訓練準確率: {train_acc}, 測試準確率: {test_acc}')

    final_test_acc = accuracy(params, x_test, y_test)
    print(f'最終測試準確率: {final_test_acc}')