# K近邻分类器 (K-Nearest Neighbors Classifier)
# 基于距离的非参数分类，y = mode(y_{1..K}) 惰性学习

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, plot_decision_boundary, save_and_close)


# ───────────────────────────── 手写 KNN 分类器 ─────────────────────────────

class KNNClassifierFromScratch:
    """从零实现 KNN 分类器"""
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """KNN 是惰性学习，只需存储训练数据"""
        self.X_train = X
        self.y_train = y
        return self

    def _distance(self, X1, X2):
        """计算样本间距离矩阵"""
        if self.metric == 'euclidean':
            return np.sqrt(((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2).sum(axis=2))
        elif self.metric == 'manhattan':
            return np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]).sum(axis=2)
        else:
            return np.sqrt(((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2).sum(axis=2))

    def predict(self, X):
        """预测每个样本的最近邻标签"""
        dists = self._distance(X, self.X_train)
        # 取 K 个最近邻的索引
        knn_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        # 投票
        knn_labels = self.y_train[knn_indices]
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=knn_labels
        )
        return predictions

    def predict_proba(self, X):
        """返回类别概率（近邻投票比例）"""
        dists = self._distance(X, self.X_train)
        knn_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        knn_labels = self.y_train[knn_indices]
        n_classes = len(np.unique(self.y_train))

        proba = np.zeros((X.shape[0], n_classes))
        for i in range(n_classes):
            proba[:, i] = (knn_labels == i).sum(axis=1) / self.n_neighbors

        return proba


def knn_classifier():
    """KNN 分类器实现（K值敏感性分析）"""
    print("K近邻分类器 (KNN) 运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=600, n_features=2,
                                         n_informative=2, n_redundant=0,
                                         n_classes=3, random_state=42)

    # 标准化（距离敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}")

    # 2. K 值敏感性分析
    print("2. K 值敏感性分析（K=1~20）...")
    k_range = range(1, 21)
    train_accs, test_accs = [], []

    for k in k_range:
        model = KNNClassifierFromScratch(n_neighbors=k, metric='euclidean')
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_accs.append(accuracy_score(y_train, train_pred))
        test_accs.append(accuracy_score(y_test, test_pred))

    best_k = k_range[np.argmax(test_accs)]
    print(f"   最优 K = {best_k} (测试集准确率={max(test_accs):.4f})")

    # 3. 最优 K 训练 & 可视化
    print(f"3. 训练最优 K={best_k} 模型...")
    model = KNNClassifierFromScratch(n_neighbors=best_k, metric='euclidean')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, cm = print_classification_report(y_test, y_pred, f"KNN (K={best_k})")

    # 4. 不同距离度量对比
    print("4. 不同距离度量对比...")
    metrics = ['euclidean', 'manhattan']
    metric_accs = {}
    for metric in metrics:
        m = KNNClassifierFromScratch(n_neighbors=best_k, metric=metric)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        metric_accs[metric] = accuracy_score(y_test, pred)
        print(f"   {metric:12s}: {metric_accs[metric]:.4f}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：K 值 vs 准确率曲线 ──
    ax = axes[0]
    ax.plot(k_range, train_accs, 'bo-', label='Train Acc', linewidth=1.5, markersize=6)
    ax.plot(k_range, test_accs, 'rs-', label='Test Acc', linewidth=1.5, markersize=6)
    ax.axvline(x=best_k, color='green', linestyle='--', linewidth=2,
               label=f'Best K={best_k}')
    ax.set_title('KNN：K 值敏感性分析')
    ax.set_xlabel('Number of Neighbors (K)')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：决策边界（前2个特征）──
    ax = axes[1]
    plot_decision_boundary(ax, X_test, y_test,
                        lambda X: model.predict(X),
                        f'KNN 决策边界\n(K={best_k}, Acc={accuracy:.3f})')

    # ── 子图3：距离度量对比 ──
    ax = axes[2]
    metric_names = list(metric_accs.keys())
    accs = list(metric_accs.values())
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(metric_names, accs, color=colors, alpha=0.85,
                edgecolor='white', width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('不同距离度量准确率对比')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    save_path = get_results_path('knn_classifier_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    knn_classifier()
