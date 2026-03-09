# SVM模型
# 可独立运行的SVM实现

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

# 生成模拟数据
def generate_data():
    """生成SVM的模拟数据"""
    # 设置随机种子，保证结果可重现
    np.random.seed(42)
    
    # 生成二分类数据，100个样本，2个特征，线性不可分
    X, y = make_classification(
        n_samples=100,  # 样本数
        n_features=2,   # 特征数
        n_informative=2,  # 有效特征数
        n_redundant=0,   # 冗余特征数
        n_classes=2,     # 类别数
        n_clusters_per_class=1,  # 每个类的簇数
        random_state=42  # 随机种子
    )
    
    return X, y

def svm():
    """SVM模型实现"""
    print("SVM模型运行中...")
    
    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 2. 模型训练
    print("2. 训练模型...")
    # 使用线性核函数
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    
    # 3. 模型预测
    print("3. 模型预测...")
    y_pred = model.predict(X_test)
    
    # 4. 模型评估
    print("4. 模型评估...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)
    
    # 分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 5. 可视化结果
    print("5. 可视化结果...")
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # 绘制支持向量
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary and Support Vectors')
    plt.grid(True)
    plt.savefig('results/svm_result.png')
    print("可视化结果已保存为 results/svm_result.png")

if __name__ == "__main__":
    svm()