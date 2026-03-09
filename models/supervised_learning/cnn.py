# CNN模型
# 可独立运行的CNN实现

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 加载数据
def load_data():
    """加载MNIST数据集"""
    # 加载MNIST数据集
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 数据预处理
    # 调整数据形状为 (样本数, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # 归一化数据
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # 将标签转换为one-hot编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

def cnn():
    """CNN模型实现"""
    print("CNN模型运行中...")
    print("开始加载数据...")
    
    # 1. 数据准备
    print("1. 准备数据...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"数据形状: {X_train.shape[1:]}")
    print("数据加载完成!")
    
    # 2. 模型构建
    print("2. 构建模型...")
    model = Sequential([
        # 第一个卷积层
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        
        # 第二个卷积层
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # 第三个卷积层
        Conv2D(64, (3, 3), activation='relu'),
        
        # 扁平化
        Flatten(),
        
        # 全连接层
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 10个类别
    ])
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 打印模型摘要
    model.summary()
    
    # 3. 模型训练
    print("3. 训练模型...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    
    # 4. 模型评估
    print("4. 模型评估...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"测试准确率: {test_acc:.4f}")
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("混淆矩阵:")
    print(cm)
    
    # 分类报告
    print("分类报告:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # 5. 可视化结果
    print("5. 可视化结果...")
    
    # 绘制训练过程
    plt.figure(figsize=(12, 4))
    
    # 准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # 损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/cnn_training_history.png')
    print("训练过程可视化结果已保存为 results/cnn_training_history.png")
    
    # 可视化一些预测结果
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(f"预测: {y_pred_classes[i]}, 真实: {y_true_classes[i]}")
    
    plt.tight_layout()
    plt.savefig('results/cnn_predictions.png')
    print("预测结果可视化已保存为 results/cnn_predictions.png")

if __name__ == "__main__":
    cnn()