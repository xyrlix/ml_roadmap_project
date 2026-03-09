# 线性回归模型
# 可独立运行的线性回归实现

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成模拟数据
def generate_data():
    """生成线性回归的模拟数据"""
    # 设置随机种子，保证结果可重现
    np.random.seed(42)
    
    # 生成100个样本
    X = np.linspace(0, 10, 100).reshape(-1, 1)  # 特征：0到10之间的100个数
    y = 2 * X + 1 + np.random.normal(0, 1, size=X.shape)  # 标签：y = 2x + 1 + 噪声
    
    return X, y

def linear_regression():
    """线性回归模型实现"""
    print("线性回归模型运行中...")
    
    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 2. 模型训练
    print("2. 训练模型...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 输出模型参数
    print(f"模型系数: {model.coef_[0][0]:.4f}")
    print(f"模型截距: {model.intercept_[0]:.4f}")
    print(f"拟合方程: y = {model.coef_[0][0]:.4f}x + {model.intercept_[0]:.4f}")
    
    # 3. 模型预测
    print("3. 模型预测...")
    y_pred = model.predict(X_test)
    
    # 4. 模型评估
    print("4. 模型评估...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"R² 评分: {r2:.4f}")
    
    # 5. 可视化结果
    print("5. 可视化结果...")
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Original Data')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Result')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/linear_regression_result.png')
    print("可视化结果已保存为 results/linear_regression_result.png")

if __name__ == "__main__":
    linear_regression()