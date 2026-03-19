# CART 决策树 (Classification and Regression Tree)
# 最小化 Gini 系数（分类）/ MSE（回归），二分分裂

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, plot_decision_boundary, save_and_close)


# ───────────────────────────── 手写 CART 决策树 ─────────────────────────────

class TreeNode:
    """决策树节点"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature_idx = feature_idx    # 分裂特征索引
        self.threshold = threshold          # 分裂阈值
        self.left = left                  # 左子树
        self.right = right                # 右子树
        self.value = value                # 叶节点预测值
        self.class_counts = class_counts    # 每类样本数（用于统计）


class CARTDecisionTree:
    """从零实现 CART 决策树（Gini 二分）"""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None

    def _gini(self, y):
        """计算 Gini 系数"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """寻找最佳分裂（最小化加权 Gini）"""
        n_samples, n_features = X.shape
        if n_samples <= self.min_samples_split:
            return None, None

        best_feature, best_threshold, best_gini = None, None, float('inf')

        parent_gini = self._gini(y)

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted_gini = (left_mask.sum() * left_gini + right_mask.sum() * right_gini) / n_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # 停止条件
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            value = np.bincount(y).argmax()
            return TreeNode(value=value, class_counts=np.bincount(y, minlength=self.n_classes))

        # 最佳分裂
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            value = np.bincount(y).argmax()
            return TreeNode(value=value, class_counts=np.bincount(y, minlength=self.n_classes))

        # 分裂
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_idx=feature_idx, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
        return self

    def _predict_single(self, x, node):
        """单样本预测"""
        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def get_tree_depth(self, node=None):
        """计算树深度"""
        if node is None:
            node = self.root
        if node.value is not None:
            return 0
        return 1 + max(self.get_tree_depth(node.left), self.get_tree_depth(node.right))


def decision_tree_cart():
    """CART 决策树实现（max_depth 敏感性分析）"""
    print("CART 决策树 (Gini 分裂) 运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=500, n_features=5,
                                         n_informative=4, n_redundant=1,
                                         n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}")

    # 2. 手写模型训练
    print("2. 训练手写 CART 决策树...")
    model_scratch = CARTDecisionTree(max_depth=5, min_samples_split=10)
    model_scratch.fit(X_train, y_train)
    y_pred_scratch = model_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    tree_depth = model_scratch.get_tree_depth()
    print(f"   手写模型准确率: {acc_scratch:.4f}")
    print(f"   树深度: {tree_depth}")

    # 3. sklearn 对比
    print("3. sklearn DecisionTreeClassifier 对比...")
    model_sk = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10, random_state=42)
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"   sklearn 准确率: {acc_sk:.4f}")

    # 4. max_depth 敏感性分析
    print("4. max_depth 敏感性分析...")
    depth_range = range(1, 11)
    train_accs, test_accs = [], []

    for depth in depth_range:
        model = CARTDecisionTree(max_depth=depth, min_samples_split=10)
        model.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, model.predict(X_train)))
        test_accs.append(accuracy_score(y_test, model.predict(X_test)))

    best_depth = depth_range[np.argmax(test_accs)]
    print(f"   最优 max_depth = {best_depth}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：max_depth vs 准确率曲线 ──
    ax = axes[0]
    ax.plot(depth_range, train_accs, 'bo-', label='Train Acc', linewidth=1.5, markersize=6)
    ax.plot(depth_range, test_accs, 'rs-', label='Test Acc', linewidth=1.5, markersize=6)
    ax.axvline(x=best_depth, color='green', linestyle='--', linewidth=2,
               label=f'Best Depth={best_depth}')
    ax.set_title('CART：max_depth 敏感性分析')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：决策边界（前2个特征）──
    ax = axes[1]
    plot_decision_boundary(ax, X_test[:, :2], y_test,
                        lambda X: model_scratch.predict(X),
                        f'CART 决策边界\n(Depth=5, Acc={acc_scratch:.3f})')

    # ── 子图3：sklearn 树结构可视化 ──
    ax = axes[2]
    plot_tree(model_sk, feature_names=[f'F{i}' for i in range(X.shape[1])],
             class_names=[f'C{i}' for i in range(len(np.unique(y)))],
             filled=True, ax=ax, fontsize=7)
    ax.set_title('sklearn 决策树结构可视化')

    save_path = get_results_path('decision_tree_cart_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    decision_tree_cart()
