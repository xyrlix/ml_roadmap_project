"""
GPT — Generative Pre-trained Transformer
==========================================
从零实现 Mini-GPT（纯 NumPy），展示自回归语言模型核心思想

实现内容：
  1. 字符级（char-level）分词
  2. 因果注意力掩码（causal/autoregressive masking）
  3. 单向（Decoder-only）Transformer 块
  4. 语言模型预训练（下一个 token 预测，Next Token Prediction）
  5. 文本生成（贪心 / Top-k / Temperature 采样）
  6. 可视化：训练损失、生成样本、注意力权重
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────── 字符级词表 ────────────────────────────

CORPUS_TEXT = """
the quick brown fox jumps over the lazy dog
a stitch in time saves nine
all that glitters is not gold
to be or not to be that is the question
the early bird catches the worm
actions speak louder than words
beauty is in the eye of the beholder
every cloud has a silver lining
good things come to those who wait
haste makes waste so take your time
the pen is mightier than the sword
practice makes perfect every day
knowledge is power and power is knowledge
the more you learn the more you earn
time flies when you are having fun
look before you leap into the unknown
a picture is worth a thousand words
where there is a will there is a way
the journey of a thousand miles begins with one step
ask not what your country can do for you
"""


def build_char_vocab(text: str) -> tuple:
    chars = sorted(set(text))
    vocab  = {c: i for i, c in enumerate(chars)}
    id2c   = {i: c for c, i in vocab.items()}
    return vocab, id2c


def text_to_ids(text: str, vocab: dict) -> np.ndarray:
    return np.array([vocab.get(c, 0) for c in text], dtype=np.int32)


def ids_to_text(ids: np.ndarray, id2c: dict) -> str:
    return "".join(id2c.get(int(i), "?") for i in ids)


def make_batches(token_ids: np.ndarray, seq_len: int, batch_size: int,
                 seed: int = 42) -> list:
    """随机切分语料为 (batch_size, seq_len+1) 的序列"""
    rng = np.random.default_rng(seed)
    N = len(token_ids) - seq_len - 1
    starts = rng.choice(max(1, N), min(N, 300), replace=False)
    batches = []
    for i in range(0, len(starts) - batch_size, batch_size):
        seqs = [token_ids[s: s + seq_len + 1] for s in starts[i:i+batch_size]]
        seqs = [s for s in seqs if len(s) == seq_len + 1]
        if len(seqs) == 0:
            continue
        batches.append(np.stack(seqs))
    return batches


# ──────────────────────── 模型组件 ──────────────────────────────

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / (s + eps)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-10)


def causal_mask(seq_len: int) -> np.ndarray:
    """上三角掩码（不含对角），值为 True 的位置将被遮蔽（加 -inf）"""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


class CausalSelfAttention:
    """带因果掩码的多头自注意力"""

    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        s = np.sqrt(2 / d_model)
        self.WQ = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.WK = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.WV = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.WO = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.last_attn = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, S, D = x.shape
        H, dk   = self.n_heads, self.d_k

        def proj_split(W):
            return (x @ W).reshape(B, S, H, dk).transpose(0, 2, 1, 3)

        Q = proj_split(self.WQ)
        K = proj_split(self.WK)
        V = proj_split(self.WV)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk)
        mask   = causal_mask(S)
        scores[:, :, mask] = -1e9

        attn = softmax(scores, axis=-1)
        self.last_attn = attn[0].mean(axis=0)   # (S, S)

        out = attn @ V
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return out @ self.WO


class GPTBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.attn = CausalSelfAttention(d_model, n_heads, seed)
        s = np.sqrt(2 / d_model)
        self.W1 = rng.normal(0, s, (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = rng.normal(0, s, (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = layer_norm(x + self.attn.forward(x))
        ff = gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        return layer_norm(x + ff)


class MiniGPT:
    """
    字符级自回归语言模型（Decoder-only Transformer）
    """
    def __init__(self, vocab_size: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2,
                 seq_len: int = 24, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.d_model   = d_model
        self.seq_len   = seq_len
        self.vocab_size = vocab_size

        s = 0.02
        self.token_emb = rng.normal(0, s, (vocab_size, d_model)).astype(np.float32)
        self.pos_emb   = rng.normal(0, s, (seq_len, d_model)).astype(np.float32)
        self.blocks    = [GPTBlock(d_model, n_heads, d_model * 4, seed=seed+i)
                          for i in range(n_layers)]
        self.W_head    = rng.normal(0, s, (d_model, vocab_size)).astype(np.float32)
        self.b_head    = np.zeros(vocab_size, dtype=np.float32)

        self.train_losses = []

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (B, S) → logits (B, S, V)"""
        B, S = token_ids.shape
        x = self.token_emb[token_ids] + self.pos_emb[:S]
        for block in self.blocks:
            x = block.forward(x)
        return x @ self.W_head + self.b_head

    def loss(self, token_ids: np.ndarray) -> float:
        """下一个 token 预测的交叉熵损失"""
        inp = token_ids[:, :-1]    # (B, S)
        tgt = token_ids[:, 1:]     # (B, S)
        logits = self.forward(inp)  # (B, S, V)
        probs  = softmax(logits, axis=-1)
        B, S   = tgt.shape
        log_p  = np.log(probs[np.arange(B)[:, None],
                               np.arange(S)[None, :],
                               tgt] + 1e-10)
        return -log_p.mean()

    def train_step(self, token_ids: np.ndarray, lr: float = 0.01):
        """简化训练：仅更新输出层和嵌入（全反向传播过于复杂）"""
        B, Sp1 = token_ids.shape
        S = Sp1 - 1
        inp = token_ids[:, :-1]
        tgt = token_ids[:, 1:]

        # 前向
        emb = self.token_emb[inp] + self.pos_emb[:S]
        h   = emb
        for block in self.blocks:
            h = block.forward(h)

        logits = h @ self.W_head + self.b_head
        probs  = softmax(logits, axis=-1)

        loss = -np.log(probs[np.arange(B)[:, None],
                              np.arange(S)[None, :],
                              tgt] + 1e-10).mean()

        # 输出头梯度
        dlogits = probs.copy()
        dlogits[np.arange(B)[:, None], np.arange(S)[None, :], tgt] -= 1
        dlogits /= (B * S)

        dW_head = h.reshape(-1, self.d_model).T @ dlogits.reshape(-1, self.vocab_size)
        db_head = dlogits.reshape(-1, self.vocab_size).sum(axis=0)
        self.W_head -= lr * dW_head
        self.b_head -= lr * db_head

        # 嵌入梯度（通过残差流近似）
        dh = dlogits @ self.W_head.T  # (B, S, D)
        for b in range(B):
            for s in range(S):
                tid = inp[b, s]
                self.token_emb[tid] -= lr * 0.3 * dh[b, s]

        return float(loss)

    def fit(self, batches: list, n_epochs: int = 40, lr: float = 0.01):
        for epoch in range(n_epochs):
            ep_loss = 0.0
            for batch in batches:
                ep_loss += self.train_step(batch, lr)
            self.train_losses.append(ep_loss / len(batches))
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}  loss={self.train_losses[-1]:.4f}")

    @staticmethod
    def _top_k_sampling(probs: np.ndarray, k: int) -> int:
        top_idx  = np.argsort(probs)[-k:]
        top_probs = probs[top_idx]
        top_probs /= top_probs.sum()
        return int(np.random.choice(top_idx, p=top_probs))

    def generate(self, prompt_ids: list, max_new_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 5,
                 seed: int = 0) -> list:
        """自回归生成"""
        np.random.seed(seed)
        ctx = list(prompt_ids)
        for _ in range(max_new_tokens):
            window = ctx[-self.seq_len:]
            inp = np.array([window], dtype=np.int32)
            logits = self.forward(inp)[0, -1]   # (V,)
            logits /= max(temperature, 1e-5)
            probs   = softmax(logits)
            next_id = self._top_k_sampling(probs, top_k)
            ctx.append(next_id)
        return ctx


# ──────────────────────── 可视化 ────────────────────────────────

def plot_results(model: MiniGPT, vocab: dict, id2c: dict,
                 generated_samples: list,
                 save_path: str = "results/gpt_results.png"):
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#1a1a2e")

    # ── 1. 训练损失 ──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(model.train_losses, color="#FF6B6B", linewidth=2)
    ax1.set_title("GPT Training Loss (NTP)", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Cross-Entropy", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    for s in ax1.spines.values(): s.set_edgecolor("#444")
    ax1.grid(True, alpha=0.2)

    # ── 2. 困惑度（Perplexity）= exp(loss）──
    ax2 = fig.add_subplot(2, 3, 2)
    ppl = np.exp(np.array(model.train_losses))
    ax2.plot(ppl, color="#4ECDC4", linewidth=2)
    ax2.set_title("Perplexity (↓ better)", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("PPL", color="white")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    for s in ax2.spines.values(): s.set_edgecolor("#444")
    ax2.grid(True, alpha=0.2)

    # ── 3. 注意力模式 ──
    ax3 = fig.add_subplot(2, 3, 3)
    prompt_text = "the quick brown"
    prompt_ids  = text_to_ids(prompt_text, vocab)
    inp = np.array([prompt_ids[-model.seq_len:]], dtype=np.int32)
    model.forward(inp)
    attn = model.blocks[-1].attn.last_attn
    S    = len(prompt_ids)
    attn_vis = attn[:S, :S]
    im = ax3.imshow(attn_vis, cmap="hot", vmin=0, vmax=attn_vis.max())
    chars = [id2c.get(int(i), "?") for i in prompt_ids[:S]]
    ax3.set_xticks(range(S))
    ax3.set_yticks(range(S))
    ax3.set_xticklabels(chars, color="white", fontsize=8)
    ax3.set_yticklabels(chars, color="white", fontsize=8)
    ax3.set_title(f'Causal Attention: "{prompt_text}"',
                  color="white", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # ── 4. 生成样本展示 ──
    ax4 = fig.add_subplot(2, 1, 2)
    ax4.axis("off")
    ax4.set_facecolor("#16213e")

    text_content = "Generated Text Samples\n" + "─" * 70 + "\n"
    for i, (prompt, gen) in enumerate(generated_samples, 1):
        gen_part = gen[len(prompt):]
        text_content += f"Prompt {i}: \"{prompt}\"\n"
        text_content += f"Generated: \"{gen_part[:80]}\"\n\n"

    ax4.text(0.02, 0.95, text_content, transform=ax4.transAxes,
             fontsize=8.5, color="white", va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.8))

    # ── 5. Token 概率分布（某个 prompt 的下一 token 预测）──
    ax5 = fig.add_subplot(2, 3, (4, 4))
    test_prompt = "the quick"
    tp_ids = text_to_ids(test_prompt, vocab)
    inp2   = np.array([tp_ids[-model.seq_len:]], dtype=np.int32)
    logits = model.forward(inp2)[0, -1]
    probs  = softmax(logits / 0.8)
    top_n  = 10
    top_idx = np.argsort(probs)[-top_n:][::-1]
    top_c   = [repr(id2c.get(int(i), "?")) for i in top_idx]
    top_p   = probs[top_idx]
    ax5.barh(range(top_n), top_p[::-1], color="#45B7D1", alpha=0.8)
    ax5.set_yticks(range(top_n))
    ax5.set_yticklabels(top_c[::-1], color="white", fontsize=8)
    ax5.set_title(f'Next Token Probs after "{test_prompt}"',
                  color="white", fontsize=9, fontweight="bold")
    ax5.set_facecolor("#16213e")
    ax5.tick_params(colors="white")
    for s in ax5.spines.values(): s.set_edgecolor("#444")
    ax5.grid(True, alpha=0.2, axis="x")

    plt.suptitle("GPT — Autoregressive Decoder-only Transformer",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ──────────────────────── 主函数 ────────────────────────────────

def gpt():
    """GPT 完整流程（字符级语言模型 + 文本生成）"""
    print("=" * 60)
    print("   GPT — 自回归 Decoder-only Transformer")
    print("=" * 60)

    # ── 1. 数据准备 ──
    print("\n[1/5] 构建字符级词表...")
    text  = CORPUS_TEXT.lower().strip()
    vocab, id2c = build_char_vocab(text)
    token_ids   = text_to_ids(text, vocab)
    print(f"  字符词表大小: {len(vocab)}")
    print(f"  总 token 数:  {len(token_ids)}")

    # ── 2. 构造训练批次 ──
    SEQ_LEN = 24
    BATCH   = 16
    batches = make_batches(token_ids, SEQ_LEN, BATCH, seed=42)
    print(f"\n[2/5] 构造训练批次: {len(batches)} 个批次，序列长度={SEQ_LEN}")

    # ── 3. 初始化模型 ──
    print("\n[3/5] 初始化 Mini-GPT (2层，d_model=32，4头)...")
    model = MiniGPT(vocab_size=len(vocab), d_model=32, n_heads=4,
                    n_layers=2, seq_len=SEQ_LEN, seed=42)

    # ── 4. 训练 ──
    print("\n[4/5] 训练中...")
    model.fit(batches, n_epochs=50, lr=0.02)
    print(f"  最终 Loss: {model.train_losses[-1]:.4f}  "
          f"PPL: {np.exp(model.train_losses[-1]):.2f}")

    # ── 5. 文本生成 ──
    print("\n[5/5] 文本生成示例:")
    prompts = [
        "the quick",
        "knowledge is",
        "time flies",
        "a picture is",
    ]
    generated_samples = []
    for prompt in prompts:
        p_ids = text_to_ids(prompt, vocab)
        gen_ids = model.generate(p_ids.tolist(), max_new_tokens=60,
                                  temperature=0.8, top_k=5, seed=42)
        gen_text = ids_to_text(np.array(gen_ids), id2c)
        print(f"  Prompt: \"{prompt}\"")
        print(f"  Output: \"{gen_text[:80]}\"")
        print()
        generated_samples.append((prompt, gen_text))

    # 可视化
    print("生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(model, vocab, id2c, generated_samples,
                 save_path="results/gpt_results.png")

    print("\n[DONE] GPT 完成!")
    return model


if __name__ == "__main__":
    gpt()
