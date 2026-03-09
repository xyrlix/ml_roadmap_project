# 随机森林模型
# 可独立运行的随机森林实现

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

# 生成模拟数据
def generate_data():
    """生成随机森林的模拟数据"""
    # 设置随机种子，保证结果可重现
    np.random.seed(42)
    
    # 生成二分类数据，200个样本，10个特征
    X, y = make_classification(
        n_samples=200,  # 样本数
        n_features=10,  # 特征数
        n_informative=5,  # 有效特征数
        n_redundant=5,   # 冗余特征数
        n_classes=2,     # 类别数
        random_state=42  # 随机种子
    )
    
    return X, y

def random_forest():
    """随机森林模型实现"""
    print("随机森林模型运行中...")
    
    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 2. 模型训练
    print("2. 训练模型...")
    # 创建随机森林分类器，100棵树
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. 模型预测
    print("3. 模型预测...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
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
    
    # 5. 特征重要性
    print("5. 特征重要性...")
    feature_importance = model.feature_importances_
    print("特征重要性:")
    for i, importance in enumerate(feature_importance):
        print(f"特征 {i+1}: {importance:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.grid(True)
    plt.savefig('results/random_forest_feature_importance.png')
    print("特征重要性可视化结果已保存为 results/random_forest_feature_importance.png")

if __name__ == "__main__":
    random_forest()