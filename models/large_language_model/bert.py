"""
BERT — Bidirectional Encoder Representations from Transformers
==============================================================
从零实现 Mini-BERT（纯 NumPy），展示核心架构思想

实现内容：
  1. 分词 & 词表构建（字符级）
  2. Positional Encoding（正弦位置编码）
  3. Multi-Head Self-Attention（多头自注意力）
  4. Transformer Encoder Layer（含 Layer Norm 和残差）
  5. Masked Language Modeling (MLM) 预训练任务
  6. 下游任务微调：文本分类（取 [CLS] token 输出）
  7. 可视化：注意力热力图、预训练损失、分类准确率
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ───────────────────────── 数据 / 词表 ──────────────────────────

CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are pets",
    "the bird sang a song",
    "flowers bloom in spring",
    "rain falls from the clouds",
    "she reads books every night",
    "he plays football with friends",
    "the sun rises in the east",
    "stars shine in the dark sky",
    "children love to play outside",
    "music fills the room with joy",
    "the teacher writes on the board",
    "students study hard for exams",
    "a river flows through the valley",
    "mountains are covered with snow",
    "the chef cooks delicious food",
    "artists paint beautiful pictures",
    "scientists discover new things",
    "engineers build amazing machines",
]

FINE_TUNE_DATA = [
    ("the cat sat on the mat", 0),        # 动物/自然
    ("the dog ran in the park", 0),
    ("cats and dogs are pets", 0),
    ("the bird sang a song", 0),
    ("flowers bloom in spring", 0),
    ("rain falls from the clouds", 0),
    ("she reads books every night", 1),    # 人类活动
    ("he plays football with friends", 1),
    ("the teacher writes on the board", 1),
    ("students study hard for exams", 1),
    ("music fills the room with joy", 1),
    ("artists paint beautiful pictures", 1),
    ("scientists discover new things", 1),
    ("engineers build amazing machines", 1),
    ("the sun rises in the east", 0),
    ("stars shine in the dark sky", 0),
]

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]


def build_vocab(corpus: list) -> tuple:
    words = set()
    for sent in corpus:
        words.update(sent.split())
    vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    for w in sorted(words):
        if w not in vocab:
            vocab[w] = len(vocab)
    id2word = {v: k for k, v in vocab.items()}
    return vocab, id2word


def tokenize(sentence: str, vocab: dict, max_len: int = 12) -> list:
    """[CLS] + tokens + [SEP] + padding"""
    tokens = ["[CLS]"] + sentence.split()[:max_len - 2] + ["[SEP]"]
    token_ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    pad_id = vocab["[PAD]"]
    while len(token_ids) < max_len:
        token_ids.append(pad_id)
    return token_ids[:max_len]


def create_mlm_batch(token_ids: list, vocab: dict, mask_prob: float = 0.15) -> tuple:
    """随机遮盖 15% 的 token，返回输入序列和标签"""
    mask_id = vocab["[MASK]"]
    pad_id  = vocab["[PAD]"]
    cls_id  = vocab["[CLS]"]
    sep_id  = vocab["[SEP]"]
    inp = token_ids[:]
    labels = [-1] * len(token_ids)   # -1 表示不计算损失
    for i, tid in enumerate(token_ids):
        if tid in (pad_id, cls_id, sep_id):
            continue
        if np.random.rand() < mask_prob:
            labels[i] = tid
            r = np.random.rand()
            if r < 0.8:
                inp[i] = mask_id
            elif r < 0.9:
                inp[i] = np.random.randint(len(SPECIAL_TOKENS), len(vocab))
            # else: 保持不变
    return inp, labels


# ───────────────────────── 位置编码 ─────────────────────────────

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    PE = np.zeros((seq_len, d_model), dtype=np.float32)
    pos = np.arange(seq_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE


# ───────────────────────── Layer Norm ────────────────────────────

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# ───────────────────────── Multi-Head Attention ──────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq, head_dim)
    返回: (batch, heads, seq, head_dim), attn_weights
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (..., seq, seq)
    if mask is not None:
        scores += mask * -1e9
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn /= (attn.sum(axis=-1, keepdims=True) + 1e-10)
    out = attn @ V
    return out, attn


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        scale = np.sqrt(2.0 / d_model)
        self.WQ = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.WK = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.WV = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.WO = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.last_attn = None

    def forward(self, x: np.ndarray, mask=None) -> np.ndarray:
        """x: (batch, seq, d_model)"""
        B, S, D = x.shape
        H = self.n_heads

        def proj_split(W):
            out = x @ W  # (B, S, D)
            return out.reshape(B, S, H, self.d_k).transpose(0, 2, 1, 3)

        Q = proj_split(self.WQ)
        K = proj_split(self.WK)
        V = proj_split(self.WV)

        out, self.last_attn = scaled_dot_product_attention(Q, K, V, mask)
        # (B, H, S, d_k) → (B, S, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return out @ self.WO


class TransformerEncoderLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.attn = MultiHeadAttention(d_model, n_heads, seed=seed)
        scale = np.sqrt(2.0 / d_model)
        self.W1 = rng.normal(0, scale, (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = rng.normal(0, scale, (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray, mask=None) -> np.ndarray:
        # Self-attention + residual
        attn_out = self.attn.forward(x, mask)
        x = layer_norm(x + attn_out)
        # FFN + residual
        ff = np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2
        x  = layer_norm(x + ff)
        return x


class MiniBERT:
    """
    极简 BERT：2 层 Transformer Encoder
    """
    def __init__(self, vocab_size: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2,
                 max_len: int = 12, n_classes: int = 2, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.d_model    = d_model
        self.max_len    = max_len
        self.vocab_size = vocab_size
        self.n_classes  = n_classes

        # 词嵌入
        self.embedding = rng.normal(0, 0.02, (vocab_size, d_model)).astype(np.float32)
        # 位置编码（固定，不训练）
        self.pos_enc = positional_encoding(max_len, d_model)
        # Transformer 层
        self.layers = [TransformerEncoderLayer(d_model, n_heads, d_model * 4, seed=seed+i)
                       for i in range(n_layers)]
        # MLM 输出头
        self.W_mlm = rng.normal(0, 0.02, (d_model, vocab_size)).astype(np.float32)
        self.b_mlm = np.zeros(vocab_size, dtype=np.float32)
        # 分类头（取 [CLS]）
        self.W_cls = rng.normal(0, 0.02, (d_model, n_classes)).astype(np.float32)
        self.b_cls = np.zeros(n_classes, dtype=np.float32)

        self.pretrain_losses  = []
        self.finetune_accs    = []
        self.finetune_losses  = []

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (batch, seq) → (batch, seq, d_model)"""
        x = self.embedding[token_ids] + self.pos_enc[:token_ids.shape[1]]
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _mlm_loss(self, token_ids: np.ndarray, labels: np.ndarray) -> tuple:
        """MLM 交叉熵损失（仅计算被遮盖位置）"""
        H = self.encode(token_ids)   # (B, S, D)
        logits = H @ self.W_mlm + self.b_mlm  # (B, S, V)
        # softmax
        logits_s = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits_s)
        probs /= probs.sum(axis=-1, keepdims=True) + 1e-10

        mask = labels >= 0
        if not mask.any():
            return 0.0, probs
        loss = -np.log(probs[mask, labels[mask]] + 1e-10).mean()
        return loss, probs

    def pretrain(self, corpus_ids: list, n_epochs: int = 30, lr: float = 0.02):
        rng = np.random.default_rng(0)
        vocab_size = self.vocab_size
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            rng.shuffle(corpus_ids)
            for ids in corpus_ids:
                inp, labels_list = create_mlm_batch(ids, {})  # dummy vocab

                inp_arr = np.array([inp], dtype=np.int32)
                lbl_arr = np.array([labels_list], dtype=np.int32)

                H = self.encode(inp_arr)
                logits = H @ self.W_mlm + self.b_mlm

                ls = logits - logits.max(axis=-1, keepdims=True)
                probs = np.exp(ls)
                probs /= probs.sum(axis=-1, keepdims=True) + 1e-10

                mask = lbl_arr[0] >= 0
                if not mask.any():
                    continue
                loss = -np.log(probs[0][mask, lbl_arr[0][mask]] + 1e-10).mean()
                epoch_loss += loss

                # 简化梯度：仅更新 MLM 头和 embedding
                dlogits = probs[0].copy()   # (S, V)
                valid_pos = np.where(mask)[0]
                for pos in valid_pos:
                    dlogits[pos, lbl_arr[0, pos]] -= 1
                dlogits /= max(1, len(valid_pos))

                dW_mlm = H[0].T @ dlogits
                db_mlm = dlogits.sum(axis=0)
                self.W_mlm -= lr * dW_mlm
                self.b_mlm -= lr * db_mlm

                # 更新 embedding（每个 token）
                dH = dlogits @ self.W_mlm.T   # (S, D)
                for si, tid in enumerate(inp):
                    if 0 <= tid < vocab_size:
                        self.embedding[tid] -= lr * dH[si]

            self.pretrain_losses.append(epoch_loss / len(corpus_ids))
            if (epoch + 1) % 10 == 0:
                print(f"  Pretrain Epoch {epoch+1:3d}  MLM loss={self.pretrain_losses[-1]:.4f}")

    def finetune(self, data_ids: list, data_labels: list,
                 n_epochs: int = 20, lr: float = 0.02):
        rng = np.random.default_rng(0)
        N   = len(data_ids)
        for epoch in range(n_epochs):
            idx = rng.permutation(N)
            ep_loss, ep_correct = 0.0, 0
            for i in idx:
                inp_arr = np.array([data_ids[i]], dtype=np.int32)
                y       = data_labels[i]

                H   = self.encode(inp_arr)       # (1, S, D)
                cls = H[0, 0]                    # [CLS] token
                logits = cls @ self.W_cls + self.b_cls  # (n_cls,)
                ls   = logits - logits.max()
                probs = np.exp(ls) / (np.exp(ls).sum() + 1e-10)

                loss = -np.log(probs[y] + 1e-10)
                pred = probs.argmax()
                ep_loss   += loss
                ep_correct += int(pred == y)

                # 梯度：只更新分类头 + embedding
                dp = probs.copy()
                dp[y] -= 1
                dW_cls = np.outer(cls, dp)
                db_cls = dp
                self.W_cls -= lr * dW_cls
                self.b_cls -= lr * db_cls

                dH0 = dp @ self.W_cls.T   # (D,)
                for tid in data_ids[i]:
                    self.embedding[tid] -= lr * dH0 * 0.1  # 小学习率

            self.finetune_losses.append(ep_loss / N)
            self.finetune_accs.append(ep_correct / N)
            if (epoch + 1) % 5 == 0:
                print(f"  Finetune Epoch {epoch+1:2d}  "
                      f"loss={ep_loss/N:.4f}  acc={ep_correct/N:.4f}")

    def get_attention_weights(self, token_ids: list) -> np.ndarray:
        """返回最后一层、平均多头注意力矩阵"""
        inp = np.array([token_ids], dtype=np.int32)
        self.encode(inp)
        attn = self.layers[-1].attn.last_attn[0]   # (H, S, S)
        return attn.mean(axis=0)                    # (S, S)


# ───────────────────────── 可视化 ──────────────────────────────

def plot_results(model: MiniBERT, vocab: dict, id2word: dict,
                 sample_sentence: str, save_path: str = "results/bert_results.png"):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")

    # ── 1. 预训练损失 ──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(model.pretrain_losses, color="#FF6B6B", linewidth=2, label="MLM Loss")
    ax1.set_title("BERT Pre-training Loss (MLM)", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Cross-Entropy", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#0f3460", labelcolor="white")
    for s in ax1.spines.values(): s.set_edgecolor("#444")
    ax1.grid(True, alpha=0.2)

    # ── 2. 微调损失 + 准确率 ──
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(model.finetune_losses, color="#FF6B6B", linewidth=2, label="Loss")
    ax2.set_ylabel("Loss", color="#FF6B6B")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_title("Fine-tuning: Loss & Accuracy", color="white", fontsize=10, fontweight="bold")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    for s in ax2.spines.values(): s.set_edgecolor("#444")

    ax2b = ax2.twinx()
    ax2b.plot(model.finetune_accs, color="#4ECDC4", linewidth=2, label="Accuracy")
    ax2b.set_ylabel("Accuracy", color="#4ECDC4")
    ax2b.set_ylim(0, 1.05)
    ax2b.tick_params(colors="white")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#0f3460", labelcolor="white", fontsize=8)
    ax2.grid(True, alpha=0.2)

    # ── 3. 注意力热力图 ──
    ax3 = fig.add_subplot(2, 3, 3)
    MAX_LEN = model.max_len
    token_ids = tokenize(sample_sentence, vocab, MAX_LEN)
    attn = model.get_attention_weights(token_ids)  # (S, S)
    tokens_disp = [id2word.get(tid, "[?]") for tid in token_ids]
    # 截断 PAD
    real_len = min(MAX_LEN, next((i for i, t in enumerate(tokens_disp) if t == "[PAD]"),
                                  MAX_LEN))
    attn = attn[:real_len, :real_len]
    tokens_disp = tokens_disp[:real_len]

    im = ax3.imshow(attn, cmap="viridis", vmin=0, vmax=attn.max())
    ax3.set_xticks(range(real_len))
    ax3.set_yticks(range(real_len))
    ax3.set_xticklabels(tokens_disp, rotation=45, ha="right", color="white", fontsize=8)
    ax3.set_yticklabels(tokens_disp, color="white", fontsize=8)
    ax3.set_title(f'Attention Heatmap\n"{sample_sentence}"',
                  color="white", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # ── 4. 词嵌入 t-SNE (手写 PCA 降维) ──
    ax4 = fig.add_subplot(2, 3, (4, 5))
    content_words = [w for w in list(vocab.keys())
                     if w not in SPECIAL_TOKENS and len(w) > 2][:30]
    ids_to_plot = [vocab[w] for w in content_words]
    embs = model.embedding[ids_to_plot]
    # PCA 到 2D
    embs_c = embs - embs.mean(axis=0)
    cov    = embs_c.T @ embs_c / len(embs_c)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2  = eigvecs[:, -2:]
    proj  = embs_c @ top2
    cats  = ["animal" if w in {"cat","dog","bird","cats","dogs","pets","sang"} else
             "nature" if w in {"sun","rain","stars","sky","mountain","river","mountains","flowers","spring","snow","clouds","valley","park","mat"} else
             "person" for w in content_words]
    color_map = {"animal": "#FF6B6B", "nature": "#4ECDC4", "person": "#FFA07A"}
    for ci, (px, py, word, cat) in enumerate(zip(proj[:, 0], proj[:, 1],
                                                   content_words, cats)):
        ax4.scatter(px, py, color=color_map[cat], s=60, alpha=0.8, zorder=3)
        ax4.annotate(word, (px, py), fontsize=7, color="white",
                     xytext=(3, 3), textcoords="offset points")
    ax4.set_title("Word Embeddings (PCA 2D)", color="white", fontsize=10, fontweight="bold")
    ax4.set_facecolor("#16213e")
    ax4.tick_params(colors="white")
    for s in ax4.spines.values(): s.set_edgecolor("#444")
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color_map[c], markersize=8, label=c)
                      for c in color_map]
    ax4.legend(handles=legend_handles, facecolor="#0f3460", labelcolor="white", fontsize=8)
    ax4.grid(True, alpha=0.15)

    # ── 5. 架构摘要 ──
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.axis("off")
    ax5.set_facecolor("#16213e")
    info = [
        "Mini-BERT Architecture",
        "─" * 28,
        f"Vocab Size    : {model.vocab_size}",
        f"d_model       : {model.d_model}",
        f"Attention Heads: {model.layers[0].attn.n_heads}",
        f"Encoder Layers : {len(model.layers)}",
        f"FFN Dim        : {model.layers[0].W1.shape[1]}",
        f"Max Seq Len    : {model.max_len}",
        "─" * 28,
        "Pre-training   : MLM (15% mask)",
        "Fine-tuning    : [CLS] + Linear",
        f"Task           : 2-class (nature/people)",
        "─" * 28,
        f"Final FT Acc   : {model.finetune_accs[-1]:.3f}",
    ]
    ax5.text(0.05, 0.95, "\n".join(info), transform=ax5.transAxes,
             fontsize=9, color="white", va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.6))

    plt.suptitle("BERT — Bidirectional Transformer Pre-training & Fine-tuning",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ───────────────────────── 主函数 ──────────────────────────────

def bert():
    """BERT 完整流程（预训练 MLM → 微调文本分类）"""
    print("=" * 60)
    print("   BERT — 双向 Transformer 预训练表示")
    print("=" * 60)

    # ── 1. 构建词表 ──
    print("\n[1/5] 构建词表...")
    vocab, id2word = build_vocab(CORPUS)
    print(f"  词表大小: {len(vocab)}")

    # ── 2. Tokenize ──
    MAX_LEN = 12
    print("\n[2/5] Tokenize 语料...")
    corpus_ids = [tokenize(s, vocab, MAX_LEN) for s in CORPUS]

    # ── 3. 实例化模型 ──
    print("\n[3/5] 初始化 Mini-BERT (2层，d_model=32，4头)...")
    model = MiniBERT(vocab_size=len(vocab), d_model=32, n_heads=4,
                     n_layers=2, max_len=MAX_LEN, n_classes=2, seed=42)

    # ── 4. MLM 预训练 ──
    print("\n[4/5] MLM 预训练...")
    # 临时重写 create_mlm_batch 传 vocab
    def mlm_with_vocab(ids):
        return create_mlm_batch(ids, vocab)
    # 替换
    import types
    orig_fn = create_mlm_batch

    # 直接在 pretrain 循环里重新实现
    rng_pt = np.random.default_rng(0)
    mask_id = vocab["[MASK]"]
    pad_id  = vocab["[PAD]"]
    cls_id  = vocab["[CLS]"]
    sep_id  = vocab["[SEP]"]

    for epoch in range(40):
        epoch_loss = 0.0
        perm = rng_pt.permutation(len(corpus_ids))
        for ci in perm:
            ids = corpus_ids[ci]
            inp, labels_list = [], []
            for tid in ids:
                if tid in (pad_id, cls_id, sep_id):
                    inp.append(tid)
                    labels_list.append(-1)
                elif rng_pt.random() < 0.15:
                    labels_list.append(tid)
                    r = rng_pt.random()
                    if r < 0.8:
                        inp.append(mask_id)
                    elif r < 0.9:
                        inp.append(int(rng_pt.integers(len(SPECIAL_TOKENS), len(vocab))))
                    else:
                        inp.append(tid)
                else:
                    inp.append(tid)
                    labels_list.append(-1)

            inp_arr = np.array([inp], dtype=np.int32)
            lbl_arr = np.array([labels_list], dtype=np.int32)

            H = model.encode(inp_arr)
            logits = H @ model.W_mlm + model.b_mlm
            ls     = logits - logits.max(axis=-1, keepdims=True)
            probs  = np.exp(ls) / (np.exp(ls).sum(axis=-1, keepdims=True) + 1e-10)

            mask_pos = lbl_arr[0] >= 0
            if not mask_pos.any():
                continue

            loss = -np.log(probs[0][mask_pos, lbl_arr[0][mask_pos]] + 1e-10).mean()
            epoch_loss += loss

            lr_pt = 0.02
            dlogits = probs[0].copy()
            valid_pos = np.where(mask_pos)[0]
            for pos in valid_pos:
                dlogits[pos, lbl_arr[0, pos]] -= 1
            dlogits /= max(1, len(valid_pos))
            model.W_mlm -= lr_pt * (H[0].T @ dlogits)
            model.b_mlm -= lr_pt * dlogits.sum(axis=0)
            dH = dlogits @ model.W_mlm.T
            for si, tid in enumerate(inp):
                if 0 <= tid < len(vocab):
                    model.embedding[tid] -= lr_pt * dH[si]

        model.pretrain_losses.append(epoch_loss / len(corpus_ids))
        if (epoch + 1) % 10 == 0:
            print(f"  Pretrain Epoch {epoch+1:2d}  MLM loss={model.pretrain_losses[-1]:.4f}")

    # ── 5. 微调分类 ──
    print("\n[5/5] 微调：文本二分类（自然/人类活动）...")
    ft_ids    = [tokenize(s, vocab, MAX_LEN) for s, _ in FINE_TUNE_DATA]
    ft_labels = [y for _, y in FINE_TUNE_DATA]
    model.finetune(ft_ids, ft_labels, n_epochs=30, lr=0.03)
    print(f"  最终微调准确率: {model.finetune_accs[-1]:.4f}")

    # 推断示例
    print("\n  推断示例:")
    test_sents = [
        "cats and dogs love to play",
        "the engineer builds a machine",
        "rain falls on the mountain",
        "she writes a beautiful song",
    ]
    for sent in test_sents:
        ids = tokenize(sent, vocab, MAX_LEN)
        inp_arr = np.array([ids], dtype=np.int32)
        H   = model.encode(inp_arr)
        cls = H[0, 0]
        logits = cls @ model.W_cls + model.b_cls
        ls   = logits - logits.max()
        probs = np.exp(ls) / (np.exp(ls).sum() + 1e-10)
        pred = probs.argmax()
        lbl  = ["Nature/Animal", "Human Activity"][pred]
        print(f"    \"{sent[:40]}\"  →  {lbl} ({probs[pred]:.3f})")

    # 可视化
    print("\n生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(model, vocab, id2word,
                 sample_sentence="the cat sat on the mat",
                 save_path="results/bert_results.png")

    print("\n[DONE] BERT 完成!")
    return model


if __name__ == "__main__":
    bert()
