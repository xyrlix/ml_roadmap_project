# 高斯朴素贝叶斯 (Gaussian Naive Bayes)
# 假设特征服从高斯分布，P(x_i|y) = N(μ, σ²)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, plot_decision_boundary, save_and_close)


# ───────────────────────────── 手写高斯朴素贝叶斯 ─────────────────────────────

class GaussianNBFromScratch:
    """从零实现高斯朴素贝叶斯：P(y|x) ∝ P(y) ∏ P(x_i|y)"""
    def __init__(self):
        self.class_priors_ = None      # P(y)
        self.class_means_ = None        # μ per class
        self.class_vars_ = None         # σ² per class
        self.classes_ = None

    def fit(self, X, y):
        """计算每类的先验和均值/方差"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_priors_ = np.zeros(n_classes)
        self.class_means_ = np.zeros((n_classes, n_features))
        self.class_vars_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[i] = len(X_c) / len(X)
            self.class_means_[i] = X_c.mean(axis=0)
            # 添加小常数避免数值不稳定
            self.class_vars_[i] = X_c.var(axis=0, ddof=1) + 1e-9

        return self

    def _gaussian_log_pdf(self, x, mu, var):
        """log N(x; μ, σ²)"""
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu)**2 / var

    def predict_log_proba(self, X):
        """计算每类 log P(y|x)"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # log P(y) + Σ log P(x_i|y)
            log_y = np.log(self.class_priors_[i] + 1e-10)
            log_likelihood = self._gaussian_log_pdf(
                X, self.class_means_[i], self.class_vars_[i]
            ).sum(axis=1)
            log_probs[:, i] = log_y + log_likelihood

        return log_probs

    def predict(self, X):
        """返回 argmax_y P(y|x)"""
        log_probs = self.predict_log_proba(X)
        return self.classes_[log_probs.argmax(axis=1)]


def gaussian_nb():
    """高斯朴素贝叶斯分类器实现"""
    print("高斯朴素贝叶斯 (Gaussian NB) 运行中...\n")

    # 1. 数据准备（连续特征）
    print("1. 准备数据（连续特征）...")
    X, y = generate_classification_data(n_samples=800, n_features=5,
                                         n_informative=4, n_redundant=1,
                                         n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}")

    # 2. 手写模型训练
    print("2. 训练手写高斯 NB...")
    model_scratch = GaussianNBFromScratch()
    model_scratch.fit(X_train, y_train)
    y_pred_scratch = model_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    print(f"   手写模型准确率: {acc_scratch:.4f}")

    # 3. sklearn 对比
    print("3. sklearn GaussianNB 对比...")
    model_sk = GaussianNB()
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"   sklearn 准确率: {acc_sk:.4f}")

    # 4. 类条件概率可视化（前2个特征）
    print("4. 可视化类条件概率分布...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：决策边界（前2个特征）──
    ax = axes[0]
    X_2d_train = X_train[:, :2]
    X_2d_test = X_test[:, :2]
    plot_decision_boundary(ax, X_2d_test, y_test,
                        lambda X: model_scratch.predict(X),
                        f'高斯 NB 决策边界\n(测试集 Acc={acc_scratch:.3f})')

    # ── 子图2：特征1的类条件分布 ──
    ax = axes[1]
    colors = ['steelblue', 'darkorange', 'seagreen']
    for i, (c, color) in enumerate(zip(model_scratch.classes_, colors)):
        X_c = X_train[y_train == c][:, 0]
        mu, var = model_scratch.class_means_[i, 0], model_scratch.class_vars_[i, 0]
        ax.hist(X_c, bins=20, alpha=0.4, color=color, density=True, label=f'Class {c}')
        x_plot = np.linspace(X_c.min()-1, X_c.max()+1, 200)
        y_plot = (1 / np.sqrt(2*np.pi*var)) * np.exp(-0.5*(x_plot-mu)**2/var)
        ax.plot(x_plot, y_plot, color=color, linewidth=2,
                label=f'μ={mu:.2f}, σ²={var:.2f}')
    ax.set_title('特征1的类条件高斯分布')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图3：混淆矩阵 ──
    ax = axes[2]
    cm = confusion_matrix(y_test, y_pred_scratch)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f'C{i}' for i in range(3)])
    ax.set_yticklabels([f'C{i}' for i in range(3)])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)
    ax.set_title('混淆矩阵（手写模型）')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    save_path = get_results_path('naive_bayes_gaussian_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    gaussian_nb()
