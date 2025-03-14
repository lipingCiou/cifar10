# MLP 用 JAX 預測 CIFAR-10

## 1. 介紹

 **如何用 JAX 訓練一個 MLP（多層感知機）來分類 CIFAR-10 圖像**。


---

## 2. 程式的主要流程

目標是建立一個 MLP 來處理 CIFAR-10 圖像分類，完整流程如下：

### 2.1 加載數據

CIFAR-10 是一個包含 **60,000 張 32x32 彩色圖片** 的數據集，這些圖片屬於 **10 個類別**（如飛機、汽車、貓等）。我們需要從 TensorFlow Datasets 讀取這些圖像，並將其轉換為 NumPy 陣列，以便使用 JAX 進行計算。

### 2.2 數據預處理（標準化）

原始圖像的像素值範圍是 **0 到 255**，我們需要將它們轉換為 **均值為 0，標準差為 1** 的標準化數據。這樣可以：
- 讓模型更容易學習特徵。
- 讓數據的不同通道（R、G、B）影響力一致。

### 2.3 初始化 MLP 網路參數

我們將建立一個三層的 MLP：
1. **輸入層（3072 維度）**：每張圖片有 32x32x3 = 3072 個像素。
2. **隱藏層 1（1024 個神經元）**，使用 ReLU 激活函數。
3. **隱藏層 2（512 個神經元）**，同樣使用 ReLU。
4. **輸出層（10 個神經元）**，對應於 10 個類別。

### 2.4 訓練流程：前向傳播（Forward Propagation）

MLP 依序計算每層的輸出：

1. 計算第一個隱藏層的輸出：
   ```math
   h_1 = ReLU(W_1 x + b_1)
   ```

2. 計算第二個隱藏層的輸出：
   ```math
   h_2 = ReLU(W_2 h_1 + b_2)
   ```

3. 計算最終輸出層（logits）：
   ```math
   y = W_{	ext{output}} h_2 + b_{	ext{output}}
   ```

### 2.5 Softmax 與損失函數（Loss Function）

輸出層的結果 \( y \) 是 **logits（分數）**，但我們希望得到的是**機率分佈**。為此，我們使用 **Softmax 函數** 來轉換：

```math
\hat{y}_i = rac{e^{y_i}}{\sum_{j=1}^{10} e^{y_j}}
```

### 2.6 訓練流程：反向傳播與 Epoch 機制

在訓練過程中，每次處理一個 **Batch**（例如 128 張圖片），執行：
1. **前向傳播（Forward Pass）**：計算預測值與 Loss。
2. **反向傳播（Backward Pass）**：透過 **鏈式法則（Chain Rule）** 計算每一層權重的梯度。
3. **更新權重**：使用梯度下降法（Gradient Descent）來調整模型的 \( W \) 和 \( b \)。

當所有批次都處理完後，**這一輪（Epoch）就結束**，然後進入下一輪。

### 2.7 反向傳播的數學推導（Chain Rule）

反向傳播的核心在於**鏈式法則（Chain Rule）**，它用來計算 **Loss 對每層權重的影響**。

假設我們的損失函數是 \( L \)，並且輸出層的預測值由隱藏層的輸出 \( h_2 \) 決定，隱藏層又由更前一層決定，我們可以透過鏈式法則來計算梯度：

$$
rac{\partial L}{\partial W_2} = rac{\partial L}{\partial y} \cdot rac{\partial y}{\partial h_2} \cdot rac{\partial h_2}{\partial W_2}
$$

$$
rac{\partial L}{\partial W_1} = rac{\partial L}{\partial y} \cdot rac{\partial y}{\partial h_2} \cdot rac{\partial h_2}{\partial h_1} \cdot rac{\partial h_1}{\partial W_1}
$$

這表示 **每一層的梯度是前面所有層的偏導數相乘**

### 2.8 訓練停止條件

訓練會持續進行，直到符合以下條件之一：
1. **達到最大 Epoch 數**（例如 10 Epochs）。
2. **Loss 停止下降**（即模型收斂）。
3. **驗證集準確率不再提升**（避免過擬合）。

---
