# GAT 图注意力网络模型
# 通过可学习注意力权重对邻居特征加权聚合，增强消息传递的表达能力

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
# 纯 numpy 实现单头 GAT 层
# 注意力系数：e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
# α_{ij} = softmax_j(e_{ij})
# h'_i = σ(Σ_j α_{ij} W h_j)
# ─────────────────────────────────────────────────────────────

def leaky_relu(x, alpha=0.2):
    return np.where(x >= 0, x, alpha * x)


def leaky_relu_grad(x, alpha=0.2):
    return np.where(x >= 0, 1.0, alpha)


def softmax_rows(X):
    e = np.exp(X - X.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-10)


def softmax_vec(x):
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-10)


class GATLayer:
    """单头 GAT 层（多头可通过叠加多个实例并 concat/mean 实现）"""
    def __init__(self, in_dim, out_dim, activation=None, seed=42):
        rng = np.random.default_rng(seed)
        std_w = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = rng.normal(0, std_w, (in_dim, out_dim))  # (in, out)
        self.a = rng.normal(0, 0.1, (2 * out_dim, 1))     # 注意力向量
        self.activation = activation

    def forward(self, A, H):
        """
        A: (N, N) 邻接矩阵（含自环）
        H: (N, in_dim)
        返回 H_out: (N, out_dim), attn: (N, N)
        """
        N = H.shape[0]
        Wh = H @ self.W          # (N, out_dim)

        # 计算注意力打分 e_{ij}
        Wh_i = np.repeat(Wh, N, axis=0)                  # (N*N, out_dim)
        Wh_j = np.tile(Wh, (N, 1))                       # (N*N, out_dim)
        pair = np.hstack([Wh_i, Wh_j])                   # (N*N, 2*out_dim)
        e = leaky_relu(pair @ self.a).reshape(N, N)       # (N, N)

        # mask：只保留边存在的位置
        mask = (A > 0).astype(float)
        e_masked = e * mask + (1 - mask) * (-1e9)

        # softmax 归一化
        attn = np.zeros((N, N))
        for i in range(N):
            row_mask = mask[i] > 0
            if row_mask.any():
                attn[i, row_mask] = softmax_vec(e_masked[i, row_mask])

        self._Wh = Wh
        self._attn = attn

        H_out = attn @ Wh   # (N, out_dim)
        if self.activation:
            return self.activation(H_out), attn
        return H_out, attn

    def update_weights(self, A, H, grad_out, lr=0.01):
        """简单梯度步"""
        dW = H.T @ (self._attn.T @ grad_out)
        self.W -= lr * dW


# ─────────────────────────────────────────────────────────────
# 多头 GAT（K 个独立头，输出 concat）
# ─────────────────────────────────────────────────────────────
class MultiHeadGAT:
    def __init__(self, in_dim, out_dim, n_heads=4,
                 n_classes=4, lr=0.01, seed=42):
        self.heads = [
            GATLayer(in_dim, out_dim, activation=leaky_relu, seed=seed + k)
            for k in range(n_heads)
        ]
        concat_dim = n_heads * out_dim
        rng = np.random.default_rng(seed + 100)
        # 输出分类层
        std = np.sqrt(2.0 / (concat_dim + n_classes))
        self.W_out = rng.normal(0, std, (concat_dim, n_classes))
        self.lr = lr
        self.n_heads = n_heads

    def _add_self_loops(self, A):
        return A + np.eye(A.shape[0])

    def forward(self, A, H):
        A_hat = self._add_self_loops(A)
        head_outs = []
        attns = []
        for head in self.heads:
            h_out, attn = head.forward(A_hat, H)
            head_outs.append(h_out)
            attns.append(attn)
        concat = np.hstack(head_outs)    # (N, n_heads * out_dim)
        logits = concat @ self.W_out     # (N, n_classes)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / (e.sum(axis=1, keepdims=True) + 1e-10)
        return probs, concat, attns

    def train_step(self, A, H, y_onehot, train_mask):
        probs, concat, attns = self.forward(A, H)
        # 损失
        eps = 1e-10
        loss = -np.mean(
            (np.log(np.clip(probs[train_mask], eps, 1)) * y_onehot[train_mask]
             ).sum(axis=1))

        # 反向：softmax + cross-entropy
        grad = probs.copy()
        grad[train_mask] -= y_onehot[train_mask]
        grad[~train_mask] = 0
        grad /= train_mask.sum()

        dW_out = concat.T @ grad
        self.W_out -= self.lr * dW_out

        dconcat = grad @ self.W_out.T   # (N, n_heads * out_dim)
        out_dim = dconcat.shape[1] // self.n_heads
        A_hat = self._add_self_loops(A)
        for k, head in enumerate(self.heads):
            dh = dconcat[:, k * out_dim:(k + 1) * out_dim]
            head.update_weights(A_hat, H, dh, lr=self.lr)
        return loss


# ─────────────────────────────────────────────────────────────
# 生成含社区结构的合成图（同 GCN）
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


def gat():
    """GAT 图注意力网络实现（合成社区图节点分类）"""
    print("GAT 图注意力网络模型运行中...\n")

    # 1. 构建图
    print("1. 构建合成社区图...")
    N_NODES, N_CLASSES = 34, 4
    A, X, labels = generate_graph(N_NODES, N_CLASSES, seed=42)
    n_edges = int(A.sum()) // 2
    print(f"   节点数: {N_NODES}  边数: {n_edges}  类别数: {N_CLASSES}")

    # 2. 训练集（每类 2 个标注节点）
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

    # 3. 训练多头 GAT（4头，隐藏维8）
    print("2. 训练多头 GAT（4 头，hidden=8）...")
    model = MultiHeadGAT(in_dim=X.shape[1], out_dim=8,
                         n_heads=4, n_classes=N_CLASSES, lr=0.03, seed=42)

    losses, train_accs, test_accs = [], [], []
    EPOCHS = 300
    for ep in range(EPOCHS):
        loss = model.train_step(A, X, y_onehot, train_mask)
        losses.append(loss)
        probs, _, _ = model.forward(A, X)
        preds = probs.argmax(axis=1)
        train_accs.append(accuracy_score(labels[train_mask], preds[train_mask]))
        test_accs.append(accuracy_score(labels[test_mask], preds[test_mask]))
        if (ep + 1) % 100 == 0:
            print(f"   Epoch {ep+1:3d}  Loss={loss:.4f}  "
                  f"Train Acc={train_accs[-1]:.3f}  Test Acc={test_accs[-1]:.3f}")

    # 4. 最终结果
    final_probs, final_emb, final_attns = model.forward(A, X)
    final_preds = final_probs.argmax(axis=1)
    test_acc = accuracy_score(labels[test_mask], final_preds[test_mask])
    print(f"\n   最终测试准确率: {test_acc:.4f}")

    # 5. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']

    # ── 子图1：训练曲线 ──
    ax = axes[0]
    ax.plot(losses, color='steelblue', lw=1.5, label='Loss')
    ax_r = ax.twinx()
    ax_r.plot(train_accs, 'g--', lw=1.5, label='Train Acc')
    ax_r.plot(test_accs, 'r:', lw=1.5, label='Test Acc')
    ax.set_title('GAT 训练曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='steelblue')
    ax_r.set_ylabel('Accuracy')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图2：注意力权重热力图（第一个头）──
    ax = axes[1]
    attn_head0 = final_attns[0]   # (N, N)
    im = ax.imshow(attn_head0, cmap='YlOrRd', aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_title('注意力权重（Head-1）')
    ax.set_xlabel('Source Node')
    ax.set_ylabel('Target Node')
    # 标注社区边界
    boundaries = [sum(labels == c) for c in range(N_CLASSES)]
    cumsum = np.cumsum(boundaries)
    node_order = np.argsort(labels)
    ax.set_xticks(cumsum[:-1])
    ax.set_yticks(cumsum[:-1])

    # ── 子图3：嵌入可视化（PCA）──
    ax = axes[2]
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(final_emb)
    for cls in range(N_CLASSES):
        mask = labels == cls
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=colors[cls], s=60, edgecolors='k',
                   linewidths=0.4, alpha=0.85, label=f'Class {cls}')
    wrong = test_mask & (final_preds != labels)
    if wrong.any():
        ax.scatter(emb_2d[wrong, 0], emb_2d[wrong, 1],
                   facecolors='none', edgecolors='black',
                   s=120, linewidths=2, zorder=5, label='Misclassified')
    ax.set_title(f'GAT 节点嵌入（PCA, Test Acc={test_acc:.3f}）')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('GAT 图注意力网络 — 节点分类（多头注意力）', fontsize=13, y=1.01)
    save_path = get_results_path('gat_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    gat()
