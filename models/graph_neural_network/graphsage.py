# GraphSAGE 图采样聚合网络模型
# 通过采样固定数量邻居并聚合（mean/max/LSTM），学习归纳式节点嵌入

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from utils import get_results_path, save_and_close


# ─────────────────────────────────────────────────────────────
# 纯 numpy 实现 GraphSAGE 层
# h'_v = σ(W · CONCAT(h_v, AGG({h_u | u ∈ N(v)})))
# ─────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)


class SAGELayer:
    """
    GraphSAGE 层，支持 mean / max 聚合器
    """
    def __init__(self, in_dim, out_dim, aggregator='mean',
                 activation=None, seed=42):
        rng = np.random.default_rng(seed)
        # W 作用于 concat(self, neighbor_agg)
        std = np.sqrt(2.0 / (2 * in_dim + out_dim))
        self.W = rng.normal(0, std, (2 * in_dim, out_dim))
        self.aggregator = aggregator
        self.activation = activation
        self._H = None
        self._concat = None

    def _aggregate(self, A, H, sample_size=10):
        """
        对每个节点采样 sample_size 个邻居，然后做 mean / max 聚合
        """
        N = H.shape[0]
        agg = np.zeros((N, H.shape[1]))
        for v in range(N):
            neighbors = np.where(A[v] > 0)[0]
            if len(neighbors) == 0:
                agg[v] = H[v]
                continue
            # 采样（有放回，fixed size）
            sampled = neighbors[
                np.random.choice(len(neighbors),
                                 min(sample_size, len(neighbors)),
                                 replace=False)
            ]
            if self.aggregator == 'mean':
                agg[v] = H[sampled].mean(axis=0)
            elif self.aggregator == 'max':
                agg[v] = H[sampled].max(axis=0)
        return agg

    def forward(self, A, H, sample_size=10):
        self._H = H
        agg = self._aggregate(A, H, sample_size)
        concat = np.hstack([H, agg])       # (N, 2*in_dim)
        self._concat = concat
        Z = concat @ self.W                # (N, out_dim)
        # L2 归一化（GraphSAGE 标准做法）
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z_norm = Z / (norm + 1e-10)
        if self.activation:
            return self.activation(Z_norm)
        return Z_norm

    def update_weights(self, grad_out, lr=0.01):
        dW = self._concat.T @ grad_out
        self.W -= lr * dW


class TwoLayerGraphSAGE:
    """两层 GraphSAGE 分类器"""
    def __init__(self, in_dim, hidden_dim, n_classes,
                 aggregator='mean', lr=0.01, seed=42):
        self.layer1 = SAGELayer(in_dim, hidden_dim, aggregator,
                                activation=relu, seed=seed)
        self.layer2 = SAGELayer(hidden_dim, n_classes, aggregator,
                                activation=None, seed=seed + 1)
        self.lr = lr

    def forward(self, A, H, sample_size=10):
        H1 = self.layer1.forward(A, H, sample_size)
        H2 = self.layer2.forward(A, H1, sample_size)
        e = np.exp(H2 - H2.max(axis=1, keepdims=True))
        probs = e / (e.sum(axis=1, keepdims=True) + 1e-10)
        return probs, H1

    def train_step(self, A, H, y_onehot, train_mask, sample_size=10):
        probs, H1 = self.forward(A, H, sample_size)

        # 损失
        eps = 1e-10
        loss = -np.mean(
            (np.log(np.clip(probs[train_mask], eps, 1)) * y_onehot[train_mask]
             ).sum(axis=1))

        # 梯度
        grad = probs.copy()
        grad[train_mask] -= y_onehot[train_mask]
        grad[~train_mask] = 0
        grad /= train_mask.sum()

        self.layer2.update_weights(grad, lr=self.lr)
        dH1 = grad @ self.layer2.W.T   # 简化反传
        dH1 = dH1 * (H1 > 0)          # ReLU mask
        self.layer1.update_weights(dH1, lr=self.lr)
        return loss


# ─────────────────────────────────────────────────────────────
# 生成含社区结构的合成图
# ─────────────────────────────────────────────────────────────
def generate_graph(n_nodes=34, n_communities=4, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array([i % n_communities for i in range(n_nodes)])
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = 0.5 if labels[i] == labels[j] else 0.05
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    noise = rng.normal(0, 0.5, (n_nodes, 8))
    X = np.hstack([deg / (deg.max() + 1e-10), noise])
    return A, X, labels


def graphsage():
    """GraphSAGE 图采样聚合网络实现（归纳式节点分类）"""
    print("GraphSAGE 图采样聚合网络模型运行中...\n")

    # 1. 构建图
    print("1. 构建合成社区图...")
    N_NODES, N_CLASSES = 34, 4
    A, X, labels = generate_graph(N_NODES, N_CLASSES, seed=42)
    n_edges = int(A.sum()) // 2
    print(f"   节点数: {N_NODES}  边数: {n_edges}  类别数: {N_CLASSES}")

    # 2. 半监督设置
    np.random.seed(42)
    train_mask = np.zeros(N_NODES, dtype=bool)
    for cls in range(N_CLASSES):
        idx = np.where(labels == cls)[0]
        chosen = np.random.choice(idx, min(2, len(idx)), replace=False)
        train_mask[chosen] = True
    test_mask = ~train_mask

    y_onehot = np.zeros((N_NODES, N_CLASSES))
    for i, c in enumerate(labels):
        y_onehot[i, c] = 1.0

    # 3. 对比 mean / max 聚合器
    print("2. 训练 GraphSAGE（mean vs max 聚合器对比）...")
    aggregators = ['mean', 'max']
    all_results = {}

    for agg in aggregators:
        model = TwoLayerGraphSAGE(
            in_dim=X.shape[1], hidden_dim=16,
            n_classes=N_CLASSES, aggregator=agg, lr=0.05, seed=42)

        losses, train_accs, test_accs = [], [], []
        EPOCHS = 300
        for ep in range(EPOCHS):
            loss = model.train_step(A, X, y_onehot, train_mask, sample_size=8)
            losses.append(loss)
            probs, _ = model.forward(A, X, sample_size=N_NODES)
            preds = probs.argmax(axis=1)
            train_accs.append(accuracy_score(labels[train_mask], preds[train_mask]))
            test_accs.append(accuracy_score(labels[test_mask], preds[test_mask]))

        final_probs, final_emb = model.forward(A, X, sample_size=N_NODES)
        final_preds = final_probs.argmax(axis=1)
        test_acc = accuracy_score(labels[test_mask], final_preds[test_mask])
        all_results[agg] = {
            'losses': losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'test_acc': test_acc,
            'emb': final_emb,
            'preds': final_preds
        }
        print(f"   [{agg}]  最终测试准确率: {test_acc:.4f}")

    # 4. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
    agg_colors = {'mean': 'steelblue', 'max': 'darkorange'}

    # ── 子图1：两种聚合器的测试准确率曲线 ──
    ax = axes[0]
    for agg in aggregators:
        ax.plot(all_results[agg]['test_accs'],
                color=agg_colors[agg], lw=1.5,
                label=f'{agg} (final={all_results[agg]["test_acc"]:.3f})')
    ax.set_title('GraphSAGE 聚合器对比（Test Acc）')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：损失曲线 ──
    ax = axes[1]
    for agg in aggregators:
        ax.plot(all_results[agg]['losses'],
                color=agg_colors[agg], lw=1.5, label=f'Loss ({agg})')
    ax.set_title('GraphSAGE 训练损失曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图3：最佳聚合器嵌入可视化（mean）──
    best_agg = max(aggregators, key=lambda k: all_results[k]['test_acc'])
    ax = axes[2]
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(all_results[best_agg]['emb'])
    for cls in range(N_CLASSES):
        mask = labels == cls
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=colors[cls], s=60, edgecolors='k',
                   linewidths=0.4, alpha=0.85, label=f'Class {cls}')
    # 标注训练节点
    ax.scatter(emb_2d[train_mask, 0], emb_2d[train_mask, 1],
               c='yellow', s=200, marker='*', zorder=5,
               edgecolors='k', label='Labeled')
    wrong = test_mask & (all_results[best_agg]['preds'] != labels)
    if wrong.any():
        ax.scatter(emb_2d[wrong, 0], emb_2d[wrong, 1],
                   facecolors='none', edgecolors='black',
                   s=120, lw=2, zorder=6, label='Misclassified')
    ta = all_results[best_agg]['test_acc']
    ax.set_title(f'GraphSAGE 嵌入（{best_agg}, Test Acc={ta:.3f}）')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('GraphSAGE — 归纳式图节点分类（邻居采样聚合）',
                 fontsize=13, y=1.01)
    save_path = get_results_path('graphsage_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    graphsage()
