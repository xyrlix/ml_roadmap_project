# 通用工具模块
# 包含：绘图设置、数据处理、评估等通用功能

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合保存图片
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────────
# 1. 路径工具
# ──────────────────────────────────────────────

# 项目根目录（utils.py 所在目录）
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"


def get_results_path(filename: str) -> str:
    """返回 results/ 目录下文件的绝对路径，自动创建目录"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / filename)


# ──────────────────────────────────────────────
# 2. 绘图设置
# ──────────────────────────────────────────────

def setup_matplotlib():
    """配置 matplotlib 全局样式，支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['figure.dpi'] = 100
    plt.style.use('seaborn-v0_8-whitegrid')


# 初始化
setup_matplotlib()


def save_and_close(filepath: str):
    """保存图片并关闭 figure，防止内存泄漏"""
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=120)
    plt.close('all')
    print(f"可视化结果已保存: {filepath}")


# ──────────────────────────────────────────────
# 3. 数据处理工具
# ──────────────────────────────────────────────

def generate_classification_data(n_samples=200, n_features=10, n_informative=5,
                                  n_redundant=3, n_classes=2, random_state=42):
    """生成分类任务的模拟数据"""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )
    return X, y


def generate_regression_data(n_samples=100, noise=1.0, random_state=42):
    """生成回归任务的模拟数据 (y = 2x + 1 + noise)"""
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.normal(0, noise, n_samples)
    return X, y


def generate_clustering_data(n_samples=300, n_clusters=4, random_state=42):
    """生成聚类任务的模拟数据"""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters,
                      cluster_std=0.8, random_state=random_state)
    return X, y


# ──────────────────────────────────────────────
# 4. 评估工具
# ──────────────────────────────────────────────

def print_classification_report(y_true, y_pred, model_name="模型"):
    """打印完整分类评估报告"""
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"{model_name} 评估结果")
    print(f"{'='*50}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n混淆矩阵:\n{cm}")
    print(f"\n分类报告:\n{classification_report(y_true, y_pred)}")
    return accuracy, cm


def print_regression_report(y_true, y_pred, model_name="模型"):
    """打印回归评估报告"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n{'='*50}")
    print(f"{model_name} 评估结果")
    print(f"{'='*50}")
    print(f"均方误差  (MSE) : {mse:.4f}")
    print(f"均方根误差(RMSE): {rmse:.4f}")
    print(f"平均绝对误差(MAE): {mae:.4f}")
    print(f"R² 评分         : {r2:.4f}")
    return mse, rmse, mae, r2


# ──────────────────────────────────────────────
# 5. 通用可视化工具
# ──────────────────────────────────────────────

def plot_decision_boundary(model, X, y, title="Decision Boundary", save_path=None):
    """绘制二维分类决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    if save_path:
        save_and_close(save_path)


def plot_feature_importance(importances, title="Feature Importance", save_path=None):
    """绘制特征重要性柱状图"""
    n_features = len(importances)
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_features), importances[indices], color='steelblue', edgecolor='white')
    plt.xticks(range(n_features), [f'F{i+1}' for i in indices], rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.title(title)
    if save_path:
        save_and_close(save_path)


def plot_training_history(history, save_path=None):
    """绘制 Keras 训练历史（准确率 + 损失）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Acc', marker='o')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Val Acc', marker='s')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train Loss', marker='o', color='orange')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Val Loss', marker='s', color='red')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    if save_path:
        save_and_close(save_path)


def plot_clusters(X, labels, centers=None, title="Clustering Result", save_path=None):
    """可视化聚类结果"""
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        label_name = f'Cluster {label}' if label >= 0 else 'Noise'
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=label_name,
                    s=50, edgecolors='k', linewidths=0.5, alpha=0.8)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*',
                    s=300, zorder=10, label='Centers', edgecolors='black')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    if save_path:
        save_and_close(save_path)
