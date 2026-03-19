# K近邻回归 (K-Nearest Neighbors Regression)
# y = mean(y_{1..K})，基于距离的非参数回归

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from utils import get_results_path, save_and_close


# ───────────────────────────── 手写 KNN 回归器 ─────────────────────────────

class KNNRegressorFromScratch:
    """从零实现 KNN 回归器（加权平均）"""
    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights  # uniform 或 distance
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _distance(self, X1, X2):
        if self.metric == 'euclidean':
            return np.sqrt(((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2).sum(axis=2))
        elif self.metric == 'manhattan':
            return np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]).sum(axis=2)
        else:
            return np.sqrt(((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2).sum(axis=2))

    def predict(self, X):
        dists = self._distance(X, self.X_train)
        knn_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        knn_distances = np.take_along_axis(dists, knn_indices, axis=1)
        knn_labels = self.y_train[knn_indices]

        if self.weights == 'uniform':
            predictions = knn_labels.mean(axis=1)
        else:  # distance 加权：w_i = 1 / (d_i + ε)
            weights = 1.0 / (knn_distances + 1e-10)
            predictions = (knn_labels * weights).sum(axis=1) / weights.sum(axis=1)

        return predictions


def generate_regression_data(n_samples=500, noise=0.1, seed=42):
    """生成非线性回归数据：y = sin(x) + 噪声"""
    np.random.seed(seed)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = np.sin(X).flatten() + np.random.normal(0, noise, n_samples)
    return X, y


def knn_regressor():
    """KNN 回归器实现（K 值敏感性 + 权重对比）"""
    print("K近邻回归器 (KNN Regression) 运行中...\n")

    # 1. 数据准备
    print("1. 准备非线性回归数据（y = sin(x) + noise）...")
    X, y = generate_regression_data(n_samples=400, noise=0.15)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}")

    # 2. K 值敏感性分析
    print("2. K 值敏感性分析（K=1~15）...")
    k_range = range(1, 16)
    train_mses, test_mses = [], []

    for k in k_range:
        model = KNNRegressorFromScratch(n_neighbors=k, metric='euclidean')
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_mses.append(mean_squared_error(y_train, y_pred_train))
        test_mses.append(mean_squared_error(y_test, y_pred_test))

    best_k = k_range[np.argmin(test_mses)]
    print(f"   最优 K = {best_k} (测试集 MSE={min(test_mses):.4f})")

    # 3. 权重策略对比
    print("3. 权重策略对比（uniform vs distance）...")
    weights = ['uniform', 'distance']
    weight_results = {}
    for w in weights:
        model = KNNRegressorFromScratch(n_neighbors=best_k, metric='euclidean', weights=w)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        weight_results[w] = {'mse': mse, 'r2': r2, 'mae': mae}
        print(f"   {w:10s}: MSE={mse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：K 值 vs MSE 曲线 ──
    ax = axes[0]
    ax.plot(k_range, train_mses, 'bo-', label='Train MSE', linewidth=1.5)
    ax.plot(k_range, test_mses, 'rs-', label='Test MSE', linewidth=1.5)
    ax.axvline(x=best_k, color='green', linestyle='--', linewidth=2,
               label=f'Best K={best_k}')
    ax.set_title('KNN 回归：K 值 vs MSE')
    ax.set_xlabel('Number of Neighbors (K)')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：预测曲线（K=best_k, uniform权重）──
    ax = axes[1]
    model = KNNRegressorFromScratch(n_neighbors=best_k, metric='euclidean', weights='uniform')
    model.fit(X_train, y_train)

    X_plot = np.linspace(X_scaled.min(), X_scaled.max(), 200).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    ax.scatter(X_test, y_test, c='steelblue', alpha=0.6, s=30, label='Test Data')
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'KNN Prediction (K={best_k})')
    ax.set_title('KNN 回归预测曲线（uniform 权重）')
    ax.set_xlabel('X (Scaled)')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图3：权重策略对比（MSE / R² / MAE）──
    ax = axes[2]
    metrics = ['MSE', 'R²', 'MAE']
    mse_vals = [weight_results[w]['mse'] for w in weights]
    r2_vals = [weight_results[w]['r2'] for w in weights]
    mae_vals = [weight_results[w]['mae'] for w in weights]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_vals, width, label='uniform',
                   color='#3498db', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, r2_vals, width, label='distance',
                   color='#e74c3c', alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('权重策略性能对比')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    save_path = get_results_path('knn_regressor_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    knn_regressor()
