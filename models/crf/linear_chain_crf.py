# 线性链条件随机场 (Linear-Chain CRF)
# 序列标注模型，考虑相邻标签依赖关系

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from utils import get_results_path, save_and_close


# ───────────────────────────── 合成序列数据 ─────────────────────────────

# 简单序列标注数据（天气预测）
SEQUENCES = [
    # (观测序列, 标签序列)
    (['sunny', 'sunny', 'cloudy', 'rain', 'rain'],
     ['hot', 'hot', 'mild', 'cool', 'cool']),
    (['rain', 'rain', 'cloudy', 'sunny', 'sunny'],
     ['cool', 'cool', 'mild', 'hot', 'hot']),
    (['sunny', 'cloudy', 'rain', 'rain', 'cloudy'],
     ['hot', 'mild', 'cool', 'cool', 'mild']),
    (['cloudy', 'sunny', 'sunny', 'cloudy', 'rain'],
     ['mild', 'hot', 'hot', 'mild', 'cool']),
    (['rain', 'rain', 'rain', 'cloudy', 'sunny'],
     ['cool', 'cool', 'cool', 'mild', 'hot']),
]

OBSERVATIONS = ['sunny', 'cloudy', 'rain']
LABELS = ['hot', 'mild', 'cool']
N_OBS = len(OBSERVATIONS)
N_LABELS = len(LABELS)


# ───────────────────────────── 手写线性链 CRF ─────────────────────────────

class LinearChainCRF:
    """从零实现线性链 CRF（简单条件随机场）"""
    def __init__(self, n_obs, n_labels, learning_rate=0.1, max_iter=100):
        self.n_obs = n_obs
        self.n_labels = n_labels
        self.lr = learning_rate
        self.max_iter = max_iter

        # 特征权重 w = [w_emit, w_trans]
        # w_emit: 观测-标签发射权重
        # w_trans: 标签-标签转移权重
        self.w_emit = np.random.randn(n_obs, n_labels) * 0.01
        self.w_trans = np.random.randn(n_labels, n_labels) * 0.01

    def _compute_potentials(self, obs_idx):
        """计算发射和转移势函数"""
        # 发射势: ψ_emit(y_t, x_t) = exp(w_emit[x_t, y_t])
        emit_potentials = np.exp(self.w_emit[obs_idx])

        # 转移势: ψ_trans(y_t, y_{t-1}) = exp(w_trans[y_t, y_{t-1}])
        trans_potentials = np.exp(self.w_trans)
        return emit_potentials, trans_potentials

    def _forward_backward(self, obs_indices):
        """前向-后向算法计算边际概率"""
        T = len(obs_indices)

        # 前向：α[t, y] = P(x_1..x_t, y_t)
        alpha = np.zeros((T, self.n_labels))
        alpha[0] = self._compute_potentials(obs_indices[0])[0]  # t=1, 无转移

        for t in range(1, T):
            emit, trans = self._compute_potentials(obs_indices[t])
            # α[t, y_t] = emit[y_t] * Σ_y_{t-1} trans[y_t, y_{t-1}] * α[t-1, y_{t-1}]
            for y_t in range(self.n_labels):
                alpha[t, y_t] = emit[y_t] * np.sum(trans[y_t] * alpha[t-1])

        # 后向：β[t, y] = P(x_{t+1}..x_T | y_t)
        beta = np.zeros((T, self.n_labels))
        beta[-1] = 1.0  # t=T, 无转移

        for t in range(T-2, -1, -1):
            emit, trans = self._compute_potentials(obs_indices[t+1])
            for y_t in range(self.n_labels):
                # β[t, y_t] = Σ_y_{t+1} trans[y_{t+1}, y_t] * emit[y_{t+1}] * β[t+1, y_{t+1}]
                beta[t, y_t] = np.sum(trans[:, y_t] * emit * beta[t+1])

        return alpha, beta

    def _compute_marginals(self, obs_indices):
        """计算边际概率 P(y_t | x_1..x_T)"""
        T = len(obs_indices)
        alpha, beta = self._compute_marginals(obs_indices)

        # 归一化常数 Z = α[T, :]
        Z = alpha[-1].sum()

        # 边际概率: P(y_t | x) = α[t, y_t] * β[t, y_t] / Z
        marginals = alpha * beta / Z
        return marginals

    def _compute_log_likelihood(self, obs_indices, label_indices):
        """计算对数似然（使用前向算法）"""
        T = len(obs_indices)
        alpha = np.zeros((T, self.n_labels))

        emit_0, _ = self._compute_potentials(obs_indices[0])
        alpha[0] = emit_0 * (1.0 / self.n_labels)  # 均匀初始分布

        for t in range(1, T):
            emit, trans = self._compute_potentials(obs_indices[t])
            for y_t in range(self.n_labels):
                alpha[t, y_t] = emit[y_t] * np.sum(trans[y_t] * alpha[t-1])

        # 对数似然: log P(y_1..y_T | x_1..x_T) = log α[T, y_T]
        log_likelihood = np.log(alpha[-1, label_indices[-1]] + 1e-10)
        return log_likelihood

    def fit(self, sequences, label_sequences):
        """训练 CRF（简化梯度上升）"""
        print("   训练线性链 CRF（简化梯度上升）...")

        for iteration in range(self.max_iter):
            total_grad_emit = np.zeros_like(self.w_emit)
            total_grad_trans = np.zeros_like(self.w_trans)

            # 随机梯度上升
            for obs_seq, label_seq in zip(sequences, label_sequences):
                obs_indices = [OBSERVATIONS.index(o) for o in obs_seq]
                label_indices = [LABELS.index(l) for l in label_seq]

                # 前向-后向
                alpha, beta = self._compute_marginals(obs_indices)
                marginals = alpha * beta
                marginals = marginals / marginals.sum(axis=1, keepdims=True)

                # 计算梯度（简化的伪似然损失）
                T = len(obs_indices)
                for t in range(T):
                    y_true = label_indices[t]
                    x_t = obs_indices[t]

                    # 发射梯度
                    for y in range(self.n_labels):
                        expected = marginals[t, y]
                        actual = 1.0 if y == y_true else 0.0
                        total_grad_emit[x_t, y] += actual - expected

                    # 转移梯度
                    if t > 0:
                        y_prev_true = label_indices[t-1]
                        for y in range(self.n_labels):
                            for y_prev in range(self.n_labels):
                                expected = marginals[t, y]
                                actual = 1.0 if y == y_true and y_prev == y_prev_true else 0.0
                                total_grad_trans[y, y_prev] += actual - expected

            # 更新权重
            self.w_emit += self.lr * total_grad_emit
            self.w_trans += self.lr * total_grad_trans

        return self

    def predict(self, obs_seq):
        """Viterbi 解码（最优标签序列）"""
        obs_indices = [OBSERVATIONS.index(o) for o in obs_seq]
        T = len(obs_indices)

        # Viterbi: V[t, y] = max P(y_1..y_t | x_1..x_t)
        V = np.zeros((T, self.n_labels))
        backpointers = np.zeros((T, self.n_labels), dtype=int)

        emit, _ = self._compute_potentials(obs_indices[0])
        V[0] = emit  # 均匀初始分布

        for t in range(1, T):
            emit, trans = self._compute_potentials(obs_indices[t])
            for y_t in range(self.n_labels):
                max_y_prev = 0
                max_val = -float('inf')
                for y_prev in range(self.n_labels):
                    val = trans[y_t, y_prev] * emit[y_t] * V[t-1, y_prev]
                    if val > max_val:
                        max_val = val
                        max_y_prev = y_prev
                V[t, y_t] = max_val
                backpointers[t, y_t] = max_y_prev

        # 回溯
        best_path = [np.argmax(V[-1])]
        for t in range(T-1, 0, -1):
            best_path.append(backpointers[t, best_path[-1]])

        return [LABELS[i] for i in reversed(best_path)]


def linear_chain_crf():
    """线性链 CRF 实现（序列标注）"""
    print("线性链条件随机场 (Linear-Chain CRF) 运行中...\n")

    # 1. 数据准备
    print("1. 准备序列标注数据...")
    sequences = [s for s, _ in SEQUENCES]
    label_sequences = [l for _, l in SEQUENCES]

    # 简单 train/test 分割
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, label_sequences, test_size=0.3, random_state=42)

    print(f"   训练序列数: {len(X_train)}  测试序列数: {len(X_test)}")

    # 2. 训练 CRF
    print("2. 训练线性链 CRF...")
    crf = LinearChainCRF(n_obs=N_OBS, n_labels=N_LABELS,
                         learning_rate=0.1, max_iter=100)
    crf.fit(X_train, y_train)

    # 3. 预测
    print("3. 预测测试序列...")
    y_pred_flat, y_true_flat = [], []

    for obs_seq, true_labels in zip(X_test, y_test):
        pred_labels = crf.predict(obs_seq)
        y_pred_flat.extend(pred_labels)
        y_true_flat.extend(true_labels)

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
    print(f"   准确率: {accuracy:.4f}")
    print(f"   F1 分数: {f1:.4f}")

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：示例序列标注对比 ──
    ax = axes[0]
    n_show = min(3, len(X_test))

    for i in range(n_show):
        obs = X_test[i]
        true = y_test[i]
        pred = crf.predict(obs)

        y_offset = 0.9 - i * 0.3
        x_pos = 0.1

        for t, (o, t_true, t_pred) in enumerate(zip(obs, true, pred)):
            color = 'green' if t_true == t_pred else 'red'
            ax.text(x_pos, y_offset, f'{o}\n{t_pred}',
                   color=color, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            x_pos += 0.18

        ax.text(x_pos + 0.05, y_offset, f'({"/".join(true)})',
               color='gray', fontsize=8, ha='left', va='center')

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'序列标注示例对比\n(Acc={accuracy:.3f})')
    ax.axis('off')

    # ── 子图2：混淆矩阵 ──
    ax = axes[1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=LABELS)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(N_LABELS))
    ax.set_yticks(range(N_LABELS))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_title('混淆矩阵')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # ── 子图3：每类准确率 ──
    ax = axes[2]
    per_class_acc = {}
    for i, label in enumerate(LABELS):
        mask = np.array(y_true_flat) == label
        if mask.sum() > 0:
            per_class_acc[label] = (np.array(y_pred_flat)[mask] == label).mean()
        else:
            per_class_acc[label] = 0.0

    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(LABELS, list(per_class_acc.values()), color=colors,
                alpha=0.85, edgecolor='white')
    for bar, acc in zip(bars, per_class_acc.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', fontsize=11)
    ax.set_title('每类准确率')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    save_path = get_results_path('linear_chain_crf_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    linear_chain_crf()
