# C4.5 决策树 (信息增益率分裂)
# 改进自 ID3，使用信息增益率处理多值特征

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import (generate_classification_data, get_results_path,
                   plot_decision_boundary, save_and_close)


class C45DecisionTree:
    """从零实现 C4.5 决策树（信息增益率）"""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None

    def _entropy(self, y):
        """计算熵"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _info_gain(self, y, splits):
        """信息增益"""
        parent_entropy = self._entropy(y)
        n_total = len(y)
        child_entropy = sum(len(s) / n_total * self._entropy(s) for s in splits)
        return parent_entropy - child_entropy

    def _split_info(self, y, splits):
        """分裂信息"""
        n_total = len(y)
        return -sum(len(s) / n_total * np.log2(len(s) / n_total + 1e-10) for s in splits)

    def _gain_ratio(self, y, splits):
        """信息增益率"""
        ig = self._info_gain(y, splits)
        si = self._split_info(y, splits)
        return ig / (si + 1e-10)

    def _best_split(self, X, y):
        """寻找最佳分裂（最大信息增益率）"""
        n_samples, n_features = X.shape
        if n_samples <= self.min_samples_split:
            return None, None

        best_feature, best_threshold, best_gain_ratio = None, None, -float('inf')

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                splits = [y[left_mask], y[right_mask]]
                gain_ratio = self._gain_ratio(y, splits)

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
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
            return TreeNode(value=value)

        # 最佳分裂
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            value = np.bincount(y).argmax()
            return TreeNode(value=value)

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
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])


def decision_tree_c45():
    """C4.5 决策树实现（信息增益率）"""
    print("C4.5 决策树（信息增益率）运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=500, n_features=5,
                                         n_informative=4, n_redundant=1,
                                         n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # 2. 训练
    print("2. 训练 C4.5 决策树...")
    model = C45DecisionTree(max_depth=5, min_samples_split=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   准确率: {acc:.4f}")

    # 3. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 决策边界
    ax = axes[0]
    plot_decision_boundary(ax, X_test[:, :2], y_test,
                        lambda X: model.predict(X),
                        f'C4.5 决策边界\n(Acc={acc:.3f})')

    # 训练/测试准确率 vs max_depth
    ax = axes[1]
    depth_range = range(1, 11)
    train_accs, test_accs = [], []
    for d in depth_range:
        m = C45DecisionTree(max_depth=d, min_samples_split=10)
        m.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, m.predict(X_train)))
        test_accs.append(accuracy_score(y_test, m.predict(X_test)))

    ax.plot(depth_range, train_accs, 'bo-', label='Train Acc')
    ax.plot(depth_range, test_accs, 'rs-', label='Test Acc')
    ax.set_title('C4.5：max_depth 敏感性分析')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('decision_tree_c45_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    decision_tree_c45()
