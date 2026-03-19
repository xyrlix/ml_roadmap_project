# 伯努利朴素贝叶斯 (Bernoulli Naive Bayes)
# 适用于二值特征（0/1），P(x_i|y) = Bernoulli(θ_y)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, save_and_close)


# ───────────────────────────── 手写伯努利 NB ─────────────────────────────

class BernoulliNBFromScratch:
    """从零实现伯努利朴素贝叶斯：二值特征分类"""
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize
        self.class_priors_ = None
        self.feature_probs_ = None
        self.classes_ = None

    def _binarize(self, X):
        """将连续特征二值化"""
        return (X > self.binarize).astype(int)

    def fit(self, X, y):
        """X 可为连续或二值，内部会二值化"""
        X_bin = self._binarize(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_priors_ = np.zeros(n_classes)
        self.feature_probs_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X_bin[y == c]
            # P(y) = (|D_c| + α) / (|D| + α * n_classes)
            self.class_priors_[i] = (len(X_c) + self.alpha) / (len(X) + self.alpha * n_classes)

            # P(x_i=1|y) = (count(x_i=1, y=c) + α) / (|D_c| + 2α)
            feature_ones = X_c.sum(axis=0) + self.alpha
            self.feature_probs_[i] = feature_ones / (len(X_c) + 2 * self.alpha)

        return self

    def predict_log_proba(self, X):
        """计算 log P(y|x)"""
        X_bin = self._binarize(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # log P(y) + Σ [x_i*log P(x_i=1|y) + (1-x_i)*log(1-P(x_i=1|y))]
            log_y = np.log(self.class_priors_[i] + 1e-10)
            log_p1 = np.log(self.feature_probs_[i] + 1e-10)
            log_p0 = np.log(1 - self.feature_probs_[i] + 1e-10)

            log_likelihood = (X_bin * log_p1 + (1 - X_bin) * log_p0).sum(axis=1)
            log_probs[:, i] = log_y + log_likelihood

        return log_probs

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return self.classes_[log_probs.argmax(axis=1)]

    def predict_proba(self, X):
        log_probs = self.predict_log_proba(X)
        # log-sum-exp 稳定归一化
        max_log = log_probs.max(axis=1, keepdims=True)
        log_probs_shifted = log_probs - max_log
        exp_probs = np.exp(log_probs_shifted)
        return exp_probs / exp_probs.sum(axis=1, keepdims=True)


def bernoulli_nb():
    """伯努利朴素贝叶斯（二值特征）实现"""
    print("伯努利朴素贝叶斯 (Bernoulli NB) 运行中...\n")

    # 1. 数据准备（高维稀疏特征）
    print("1. 准备数据（高维稀疏二值特征）...")
    X, y = generate_classification_data(n_samples=1000, n_features=15,
                                         n_informative=8, n_redundant=4,
                                         n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}  特征数: {X.shape[1]}")

    # 2. 手写模型训练
    print("2. 训练手写伯努利 NB...")
    model_scratch = BernoulliNBFromScratch(alpha=1.0, binarize=0.0)
    model_scratch.fit(X_train, y_train)
    y_pred_scratch = model_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    print(f"   手写模型准确率: {acc_scratch:.4f}")

    # 3. sklearn 对比
    print("3. sklearn BernoulliNB 对比...")
    model_sk = BernoulliNB(alpha=1.0, binarize=0.0)
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"   sklearn 准确率: {acc_sk:.4f}")

    # 4. ROC 曲线
    print("4. 绘制 ROC 曲线...")
    proba_scratch = model_scratch.predict_proba(X_test)[:, 1]
    proba_sk = model_sk.predict_proba(X_test)[:, 1]

    fpr_scr, tpr_scr, _ = roc_curve(y_test, proba_scratch)
    fpr_sk, tpr_sk, _ = roc_curve(y_test, proba_sk)
    auc_scr = auc(fpr_scr, tpr_scr)
    auc_sk = auc(fpr_sk, tpr_sk)

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：准确率对比 ──
    ax = axes[0]
    models = ['Scratch', 'sklearn']
    accs = [acc_scratch, acc_sk]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(models, accs, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('模型准确率对比')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图2：ROC 曲线对比 ──
    ax = axes[1]
    ax.plot(fpr_scr, tpr_scr, color='#3498db', linewidth=2,
            label=f'Scratch (AUC={auc_scr:.3f})')
    ax.plot(fpr_sk, tpr_sk, color='#e74c3c', linewidth=2, linestyle='--',
            label=f'sklearn (AUC={auc_sk:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_title('ROC 曲线对比')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图3：特征概率热力图（手写模型 Class 0）──
    ax = axes[2]
    feat_probs_class0 = model_scratch.feature_probs_[0]
    im = ax.imshow(feat_probs_class0.reshape(3, 5),
                  aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Class 0 的特征概率 P(x_i=1|y=0)\n(前15个特征热力图)')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Index')

    save_path = get_results_path('naive_bayes_bernoulli_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    bernoulli_nb()
