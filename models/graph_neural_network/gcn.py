# GCN 图卷积网络模型
# 基于谱图理论的图神经网络，通过聚合邻居特征学习节点表示

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils import get_results_path, save_and_close


# ─────────────────────────────────────────────────────────────
# 手动实现 GCN（不依赖 PyG / DGL，纯 numpy）
# 支持多层图卷积：H^(l+1) = σ(Ã H^(l) W^(l))
# Ã = D^{-1/2}(A+I)D^{-1/2}  （归一化邻接矩阵）
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(A):
    """计算对称归一化邻接矩阵 D^{-1/2}(A+I)D^{-1/2}"""
    N = A.shape[0]
    A_hat = A + np.eye(N)            # 加自环
    D = np.diag(A_hat.sum(axis=1))  # 度矩阵
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true_onehot, mask):
    """仅计算 mask=True 节点的损失"""
    eps = 1e-10
    log_pred = np.log(np.clip(y_pred[mask], eps, 1))
    return -np.mean((log_pred * y_true_onehot[mask]).sum(axis=1))


class GCNLayer:
    """单层图卷积：H_out = σ(Ã H W)"""
    def __init__(self, in_dim, out_dim, activation=None, seed=None):
        rng = np.random.default_rng(seed)
        # Xavier 初始化
        std = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = rng.normal(0, std, (in_dim, out_dim))
        self.activation = activation
        # 缓存
        self._H = None
        self._AH = None

    def forward(self, A_norm, H):
        self._H = H
        AH = A_norm @ H         # (N, in_dim)
        self._AH = AH
        Z = AH @ self.W         # (N, out_dim)
        if self.activation:
            return self.activation(Z)
        return Z

    def backward(self, A_norm, grad_out, lr=0.01):
        """简单梯度更新"""
        dW = self._AH.T @ grad_out   # (in_dim, out_dim)
        dH = A_norm @ (grad_out @ self.W.T)
        self.W -= lr * dW
        return dH


class TwoLayerGCN:
    """两层 GCN 分类器"""
    def __init__(self, in_dim, hidden_dim, n_classes, lr=0.01, seed=42):
        self.conv1 = GCNLayer(in_dim, hidden_dim, activation=relu, seed=seed)
        self.conv2 = GCNLayer(hidden_dim, n_classes, activation=None, seed=seed+1)
        self.lr = lr

    def forward(self, A_norm, X):
        H1 = self.conv1.forward(A_norm, X)
        H2 = self.conv2.forward(A_norm, H1)
        return softmax(H2), H1   # 返回概率和中间嵌入

    def train_step(self, A_norm, X, y_onehot, train_mask):
        # 前向
        probs, H1 = self.forward(A_norm, X)

        # 损失（仅训练节点）
        loss = cross_entropy_loss(probs, y_onehot, train_mask)

        # 反向（softmax + cross-entropy 联合梯度）
        grad = probs.copy()
        grad[train_mask] -= y_onehot[train_mask]
        grad[~train_mask] = 0
        grad /= train_mask.sum()

        dH1 = self.conv2.backward(A_norm, grad, lr=self.lr)
        # conv1 的激活梯度（ReLU）
        relu_mask = (H1 > 0).astype(float)
        dH1 = dH1 * relu_mask
        self.conv1.backward(A_norm, dH1, lr=self.lr)

        return loss


# ─────────────────────────────────────────────────────────────
# 生成 Karate Club 风格合成图
# ─────────────────────────────────────────────────────────────

def generate_graph(n_nodes=34, n_communities=4, seed=42):
    """
    生成含社区结构的合成图（类 Karate Club）：
    - n_nodes 个节点分为 n_communities 个社区
    - 社区内边密度高，社区间密度低
    """
    rng = np.random.default_rng(seed)
    labels = np.array([i % n_communities for i in range(n_nodes)])

    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if labels[i] == labels[j]:
                p = 0.5   # 同社区
            else:
                p = 0.05  # 跨社区
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0

    # 节点特征：one-hot 度 + 随机噪声
    deg = A.sum(axis=1, keepdims=True)
    noise = rng.normal(0, 0.5, (n_nodes, 8))
    X = np.hstack([deg / deg.max(), noise])

    return A, X, labels


def gcn():
    """GCN 图卷积网络实现（合成社区图节点分类）"""
    print("GCN 图卷积网络模型运行中...\n")

    # 1. 构建图
    print("1. 构建合成社区图...")
    N_NODES = 34
    N_CLASSES = 4
    A, X, labels = generate_graph(n_nodes=N_NODES, n_communities=N_CLASSES, seed=42)
    A_norm = normalize_adjacency(A)

    n_edges = int(A.sum()) // 2
    print(f"   节点数: {N_NODES}  边数: {n_edges}  类别数: {N_CLASSES}")
    print(f"   特征维度: {X.shape[1]}")

    # 2. 训练/测试划分（半监督：每类 2 个已标注节点）
    np.random.seed(42)
    train_mask = np.zeros(N_NODES, dtype=bool)
    for cls in range(N_CLASSES):
        idx = np.where(labels == cls)[0]
        chosen = np.random.choice(idx, min(2, len(idx)), replace=False)
        train_mask[chosen] = True
    test_mask = ~train_mask
    print(f"   已标注: {train_mask.sum()}  未标注(测试): {test_mask.sum()}")

    # One-hot 标签
    y_onehot = np.zeros((N_NODES, N_CLASSES))
    for i, c in enumerate(labels):
        y_onehot[i, c] = 1.0

    # 3. 训练 GCN
    print("2. 训练两层 GCN...")
    model = TwoLayerGCN(in_dim=X.shape[1], hidden_dim=16,
                        n_classes=N_CLASSES, lr=0.05, seed=42)

    losses, train_accs, test_accs = [], [], []
    EPOCHS = 300
    for ep in range(EPOCHS):
        loss = model.train_step(A_norm, X, y_onehot, train_mask)
        losses.append(loss)

        probs, _ = model.forward(A_norm, X)
        preds = probs.argmax(axis=1)
        train_accs.append(accuracy_score(labels[train_mask], preds[train_mask]))
        test_accs.append(accuracy_score(labels[test_mask], preds[test_mask]))

        if (ep + 1) % 100 == 0:
            print(f"   Epoch {ep+1:3d}  Loss={loss:.4f}  "
                  f"Train Acc={train_accs[-1]:.3f}  Test Acc={test_accs[-1]:.3f}")

    # 4. 最终嵌入（conv1 输出）
    final_probs, final_emb = model.forward(A_norm, X)
    final_preds = final_probs.argmax(axis=1)
    test_acc = accuracy_score(labels[test_mask], final_preds[test_mask])
    print(f"\n   最终测试准确率: {test_acc:.4f}")

    # 5. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']

    # ── 子图1：图结构（弹簧布局近似）──
    ax = axes[0]
    # 用 MDS 将节点嵌入到 2D（基于最短路径距离）
    from sklearn.manifold import MDS
    dist = np.zeros((N_NODES, N_NODES))
    for i in range(N_NODES):
        for j in range(N_NODES):
            if A[i, j] > 0:
                dist[i, j] = 1.0
            elif i != j:
                dist[i, j] = 2.0
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=42, normalized_stress='auto')
    pos = mds.fit_transform(dist)

    # 画边
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if A[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        'gray', lw=0.5, alpha=0.4, zorder=1)
    # 画节点
    for cls in range(N_CLASSES):
        mask = labels == cls
        ax.scatter(pos[mask, 0], pos[mask, 1],
                   c=colors[cls], s=80, edgecolors='k',
                   linewidths=0.5, zorder=3, label=f'Class {cls}')
    # 标注训练节点
    ax.scatter(pos[train_mask, 0], pos[train_mask, 1],
               c='yellow', s=200, marker='*', zorder=4,
               edgecolors='k', label='Labeled')
    ax.set_title('图结构（社区着色）')
    ax.legend(fontsize=7, loc='best')
    ax.axis('off')

    # ── 子图2：训练曲线 ──
    ax = axes[1]
    ax.plot(losses, color='steelblue', linewidth=1.5, label='Loss')
    ax_r = ax.twinx()
    ax_r.plot(train_accs, 'g--', linewidth=1.5, label='Train Acc')
    ax_r.plot(test_accs, 'r:', linewidth=1.5, label='Test Acc')
    ax.set_title('GCN 训练曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='steelblue')
    ax_r.set_ylabel('Accuracy')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图3：GCN 嵌入（中间层 t-SNE）──
    ax = axes[2]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(final_emb)
    for cls in range(N_CLASSES):
        mask = labels == cls
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=colors[cls], s=60, edgecolors='k',
                   linewidths=0.4, alpha=0.85, label=f'Class {cls}')
    # 标注预测错误的节点
    wrong = test_mask & (final_preds != labels)
    if wrong.any():
        ax.scatter(emb_2d[wrong, 0], emb_2d[wrong, 1],
                   facecolors='none', edgecolors='black',
                   s=120, linewidths=2, zorder=5, label='Misclassified')
    ax.set_title(f'GCN 节点嵌入（PCA, Test Acc={test_acc:.3f}）')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('GCN 图卷积网络 — 节点分类（合成社区图）', fontsize=13, y=1.01)
    save_path = get_results_path('gcn_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    gcn()
