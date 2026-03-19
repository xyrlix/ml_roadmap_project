# Word2Vec 词向量模型
# 使用 Skip-Gram + 负采样 (SGNS) 从零实现词向量学习

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from utils import get_results_path, save_and_close


# ─────────────────────────────────────────────────────────────
# 语料库（小型人工语料，模拟 NLP 语义）
# ─────────────────────────────────────────────────────────────
CORPUS = [
    "the king is a powerful man who rules the kingdom",
    "the queen is a powerful woman who rules the kingdom",
    "the prince is the son of the king and queen",
    "the princess is the daughter of the king and queen",
    "the man loves the woman and the woman loves the man",
    "the king and the queen ruled the kingdom together",
    "the dog is a loyal animal and loves his owner",
    "the cat is a cute animal but is not as loyal as the dog",
    "paris is the capital of france and london is the capital of england",
    "france and england are countries in europe",
    "the programmer writes code and the scientist does research",
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks for learning representations",
    "the neural network learns from data using gradient descent",
    "a word is represented as a vector in word2vec",
    "similar words have similar vector representations",
    "king minus man plus woman equals queen in word vector space",
    "the cat sat on the mat and the dog lay on the floor",
    "natural language processing deals with understanding text",
    "word embeddings capture semantic relationships between words",
]


def build_vocab(corpus, min_count=1):
    """构建词汇表"""
    words = []
    for sent in corpus:
        words.extend(sent.lower().split())
    counter = Counter(words)
    vocab = [w for w, c in counter.items() if c >= min_count]
    vocab = sorted(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    freq = np.array([counter[w] for w in vocab], dtype=float)
    return word2idx, idx2word, vocab, freq


def generate_skipgram_pairs(corpus, word2idx, window=2):
    """生成 (中心词, 上下文词) 训练对"""
    pairs = []
    for sent in corpus:
        tokens = [w for w in sent.lower().split() if w in word2idx]
        indices = [word2idx[w] for w in tokens]
        for i, center in enumerate(indices):
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if i != j:
                    pairs.append((center, indices[j]))
    return np.array(pairs)


class Word2Vec:
    """
    Skip-Gram + 负采样 (SGNS) 实现
    目标：最大化 log P(context | center) - k * E[log P(noise | center)]
    """
    def __init__(self, vocab_size, embed_dim=50, seed=42):
        rng = np.random.default_rng(seed)
        # 中心词嵌入矩阵（输入矩阵）
        self.W_in  = rng.uniform(-0.5/embed_dim, 0.5/embed_dim,
                                 (vocab_size, embed_dim))
        # 上下文词嵌入矩阵（输出矩阵）
        self.W_out = np.zeros((vocab_size, embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def train_pair(self, center_idx, ctx_idx, neg_indices, lr=0.025):
        """
        对单个 (center, context, [negatives]) 三元组做参数更新
        损失 = -log σ(v_ctx · v_center) - Σ log σ(-v_neg · v_center)
        """
        v_center = self.W_in[center_idx]        # (embed_dim,)

        # 正样本
        v_ctx = self.W_out[ctx_idx]
        score_pos = self._sigmoid(v_ctx @ v_center)
        grad_pos_ctx = (score_pos - 1) * v_center
        grad_center = (score_pos - 1) * v_ctx

        # 负样本
        for neg in neg_indices:
            v_neg = self.W_out[neg]
            score_neg = self._sigmoid(v_neg @ v_center)
            grad_neg_ctx = score_neg * v_center
            grad_center += score_neg * v_neg
            self.W_out[neg] -= lr * grad_neg_ctx

        # 更新
        self.W_out[ctx_idx] -= lr * grad_pos_ctx
        self.W_in[center_idx] -= lr * grad_center

    def get_embeddings(self):
        """返回平均嵌入矩阵"""
        return (self.W_in + self.W_out) / 2.0

    def similarity(self, idx1, idx2):
        E = self.get_embeddings()
        v1 = E[idx1]; v2 = E[idx2]
        return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def most_similar(self, query_idx, topn=5):
        E = self.get_embeddings()
        q = E[query_idx]
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / (norms + 1e-10)
        sims = E_norm @ q_norm
        sims[query_idx] = -2  # 排除自身
        top_idx = np.argsort(sims)[::-1][:topn]
        return list(zip(top_idx, sims[top_idx]))

    def analogy(self, a_idx, b_idx, c_idx):
        """a - b + c → d"""
        E = self.get_embeddings()
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_n = E / (norms + 1e-10)
        query = E_n[b_idx] - E_n[a_idx] + E_n[c_idx]
        query = query / (np.linalg.norm(query) + 1e-10)
        sims = E_n @ query
        for idx in [a_idx, b_idx, c_idx]:
            sims[idx] = -2
        return int(np.argmax(sims)), float(np.max(sims))


def word2vec():
    """Word2Vec Skip-Gram 负采样实现"""
    print("词向量模型运行中...\n")

    # 1. 构建语料与词表
    print("1. 构建词汇表...")
    word2idx, idx2word, vocab, freq = build_vocab(CORPUS, min_count=1)
    V = len(vocab)
    print(f"   词汇量: {V}")

    # 2. 生成训练对
    print("2. 生成 Skip-Gram 训练对 (window=2)...")
    pairs = generate_skipgram_pairs(CORPUS, word2idx, window=2)
    print(f"   训练对数量: {len(pairs)}")

    # 负采样分布：频率^(3/4)
    neg_dist = freq ** 0.75
    neg_dist = neg_dist / neg_dist.sum()

    # 3. 训练
    print("3. 训练 Word2Vec (SGNS)...")
    EMBED_DIM = 50
    EPOCHS = 200
    K_NEG = 5       # 负样本数
    LR_START = 0.025
    LR_END   = 0.001

    model = Word2Vec(V, embed_dim=EMBED_DIM, seed=42)
    losses = []

    rng = np.random.default_rng(42)
    for ep in range(EPOCHS):
        lr = LR_START - (LR_START - LR_END) * ep / EPOCHS
        idx_order = rng.permutation(len(pairs))
        ep_loss = 0
        for k in idx_order:
            center, ctx = int(pairs[k, 0]), int(pairs[k, 1])
            # 采样负样本（不含正例）
            neg_idx = rng.choice(V, size=K_NEG, p=neg_dist, replace=False)
            neg_idx = neg_idx[neg_idx != ctx][:K_NEG]
            if len(neg_idx) == 0:
                neg_idx = np.array([0])
            model.train_pair(center, ctx, neg_idx, lr=lr)

            # 近似损失
            v_c = model.W_in[center]
            v_ctx = model.W_out[ctx]
            sig = model._sigmoid(v_ctx @ v_c)
            ep_loss += -np.log(sig + 1e-10)

        losses.append(ep_loss / len(pairs))
        if (ep + 1) % 50 == 0:
            print(f"   Epoch {ep+1:3d}  Loss={losses[-1]:.4f}  lr={lr:.4f}")

    # 4. 评估类比关系
    print("\n4. 词汇类比分析...")
    E = model.get_embeddings()

    # 最近邻
    test_words = ['king', 'woman', 'france', 'learning']
    print("   最近邻词汇：")
    for w in test_words:
        if w in word2idx:
            similar = model.most_similar(word2idx[w], topn=3)
            sim_str = ", ".join(f"{idx2word[i]}({s:.2f})" for i, s in similar)
            print(f"   '{w}': {sim_str}")

    # 类比 king - man + woman = queen
    analogy_tests = [
        ('king', 'man', 'woman', 'queen'),
        ('france', 'paris', 'england', 'london'),
    ]
    print("   词汇类比：")
    for a, b, c, expected in analogy_tests:
        if all(w in word2idx for w in [a, b, c]):
            result_idx, score = model.analogy(
                word2idx[a], word2idx[b], word2idx[c])
            print(f"   {a} - {b} + {c} = {idx2word[result_idx]} "
                  f"(expected: {expected}, score={score:.3f})")

    # 5. 可视化
    print("\n5. 可视化词向量...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── 子图1：训练损失 ──
    ax = axes[0]
    ax.plot(losses, color='steelblue', lw=1.5)
    ax.set_title('Word2Vec 训练损失曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.grid(True, alpha=0.3)

    # ── 子图2：词向量 t-SNE/PCA 可视化 ──
    from sklearn.decomposition import PCA
    ax = axes[1]
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(E)

    # 选取重点词汇展示
    highlight_groups = {
        'royalty': ['king', 'queen', 'prince', 'princess'],
        'gender': ['man', 'woman'],
        'country': ['france', 'england'],
        'tech': ['machine', 'learning', 'neural', 'network'],
    }
    group_colors = {'royalty': '#e41a1c', 'gender': '#377eb8',
                    'country': '#4daf4a', 'tech': '#ff7f00'}

    for group, words in highlight_groups.items():
        for w in words:
            if w in word2idx:
                idx = word2idx[w]
                ax.scatter(emb_2d[idx, 0], emb_2d[idx, 1],
                           c=group_colors[group], s=80,
                           edgecolors='k', linewidths=0.5, zorder=5)
                ax.annotate(w, (emb_2d[idx, 0], emb_2d[idx, 1]),
                            fontsize=8, ha='center',
                            xytext=(3, 3), textcoords='offset points')

    # 图例
    for group, color in group_colors.items():
        ax.scatter([], [], c=color, s=60, label=group)
    ax.set_title('词向量 PCA 可视化')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图3：余弦相似度热力图 ──
    ax = axes[2]
    target_words = ['king', 'queen', 'man', 'woman',
                    'france', 'england', 'learning', 'network']
    target_words = [w for w in target_words if w in word2idx]
    n_tw = len(target_words)
    sim_matrix = np.zeros((n_tw, n_tw))
    E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    for i, wi in enumerate(target_words):
        for j, wj in enumerate(target_words):
            sim_matrix[i, j] = float(
                E_n[word2idx[wi]] @ E_n[word2idx[wj]])
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n_tw))
    ax.set_yticks(range(n_tw))
    ax.set_xticklabels(target_words, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(target_words, fontsize=9)
    for i in range(n_tw):
        for j in range(n_tw):
            ax.text(j, i, f'{sim_matrix[i,j]:.2f}',
                    ha='center', va='center', fontsize=7)
    ax.set_title('词向量余弦相似度矩阵')

    plt.suptitle('Word2Vec — Skip-Gram 负采样词向量学习', fontsize=13, y=1.01)
    save_path = get_results_path('word2vec_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    word2vec()
