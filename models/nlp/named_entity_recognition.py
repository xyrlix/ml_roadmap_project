"""
命名实体识别 (Named Entity Recognition, NER)
============================================
基于特征工程 + 序列标注的 NER 实现（纯 NumPy / sklearn，无需深度学习框架）

核心思路：
  1. 将 NER 建模为序列标注问题，使用 BIO 标注方案
     - B-PER / I-PER : 人名
     - B-LOC / I-LOC : 地名
     - B-ORG / I-ORG : 机构名
     - O              : 非实体
  2. 为每个词提取上下文特征（词形、前后缀、大小写、词性启发式等）
  3. 将序列标注转化为多分类，用 Logistic Regression / Linear SVC 独立预测每个位置
  4. 后处理：修正 I-* 标签不能出现在 O/B-* 不匹配之后的情况
  5. 可视化：打印带颜色的实体标注结果，并统计各类别 F1
"""

import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ─────────────────────────── 数据集构造 ─────────────────────────────

TRAIN_SENTENCES = [
    # (tokens, labels)  BIO 标注
    (["Apple", "Inc.", "was", "founded", "by", "Steve", "Jobs", "in", "California", "."],
     ["B-ORG", "I-ORG", "O", "O", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]),

    (["Elon", "Musk", "leads", "Tesla", "and", "SpaceX", "."],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "B-ORG", "O"]),

    (["The", "headquarters", "of", "Google", "is", "in", "Mountain", "View", "."],
     ["O", "O", "O", "B-ORG", "O", "O", "B-LOC", "I-LOC", "O"]),

    (["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
     ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O"]),

    (["Microsoft", "acquired", "LinkedIn", "for", "$", "26", "billion", "."],
     ["B-ORG", "O", "B-ORG", "O", "O", "O", "O", "O"]),

    (["Paris", "is", "the", "capital", "of", "France", "."],
     ["B-LOC", "O", "O", "O", "O", "B-LOC", "O"]),

    (["Jeff", "Bezos", "founded", "Amazon", "in", "Seattle", "."],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O"]),

    (["The", "United", "Nations", "is", "based", "in", "New", "York", "."],
     ["O", "B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O"]),

    (["Sundar", "Pichai", "is", "the", "CEO", "of", "Alphabet", "Inc.", "."],
     ["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG", "I-ORG", "O"]),

    (["London", "is", "the", "largest", "city", "in", "the", "United", "Kingdom", "."],
     ["B-LOC", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"]),

    (["Nike", "is", "headquartered", "in", "Beaverton", ",", "Oregon", "."],
     ["B-ORG", "O", "O", "O", "B-LOC", "O", "B-LOC", "O"]),

    (["Warren", "Buffett", "chairs", "Berkshire", "Hathaway", "."],
     ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O"]),

    (["The", "Amazon", "River", "flows", "through", "Brazil", "."],
     ["O", "B-LOC", "I-LOC", "O", "O", "B-LOC", "O"]),

    (["IBM", "was", "founded", "in", "New", "York", "in", "1911", "."],
     ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O", "O", "O"]),

    (["Satya", "Nadella", "became", "CEO", "of", "Microsoft", "in", "2014", "."],
     ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "O", "O", "O"]),

    (["The", "Eiffel", "Tower", "is", "located", "in", "Paris", "."],
     ["O", "B-LOC", "I-LOC", "O", "O", "O", "B-LOC", "O"]),

    (["Angela", "Merkel", "served", "as", "Chancellor", "of", "Germany", "."],
     ["B-PER", "I-PER", "O", "O", "O", "O", "B-LOC", "O"]),

    (["Twitter", "was", "acquired", "by", "Elon", "Musk", "in", "2022", "."],
     ["B-ORG", "O", "O", "O", "B-PER", "I-PER", "O", "O", "O"]),

    (["The", "Great", "Wall", "stretches", "across", "northern", "China", "."],
     ["O", "B-LOC", "I-LOC", "O", "O", "O", "B-LOC", "O"]),

    (["Intel", "supplies", "chips", "to", "Apple", "and", "Dell", "."],
     ["B-ORG", "O", "O", "O", "B-ORG", "O", "B-ORG", "O"]),
]

TEST_SENTENCES = [
    (["Tim", "Cook", "runs", "Apple", "from", "Cupertino", "."],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O"]),

    (["Meta", "Platforms", "is", "headquartered", "in", "Menlo", "Park", "."],
     ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O"]),

    (["Mount", "Everest", "is", "on", "the", "border", "of", "Nepal", "and", "Tibet", "."],
     ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "O", "B-LOC", "O"]),

    (["Sam", "Altman", "is", "the", "CEO", "of", "OpenAI", "."],
     ["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG", "O"]),

    (["Toyota", "manufactures", "cars", "in", "Japan", "."],
     ["B-ORG", "O", "O", "O", "B-LOC", "O"]),
]

# ─────────────────────────── 特征提取 ─────────────────────────────

LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
N_LABELS = len(LABEL2ID)

# 常用人名词典（启发式）
PERSON_NAMES = {"Steve", "Jobs", "Elon", "Musk", "Barack", "Obama", "Jeff", "Bezos",
                "Sundar", "Pichai", "Warren", "Buffett", "Satya", "Nadella", "Angela",
                "Merkel", "Tim", "Cook", "Sam", "Altman"}

ORG_SUFFIXES = {"Inc.", "Ltd.", "Corp.", "LLC", "Co.", "Holdings", "Group", "Platforms"}
LOC_WORDS    = {"City", "River", "Mountain", "Mount", "Tower", "Wall", "Kingdom",
                "County", "Island", "Ocean", "Sea", "Lake"}


def word_features(tokens: list, i: int) -> dict:
    """为位置 i 的词提取特征字典"""
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.isupper": int(word.isupper()),
        "word.istitle": int(word.istitle()),
        "word.isdigit": int(word.isdigit()),
        "word.len": len(word),
        "word.prefix2": word[:2].lower(),
        "word.prefix3": word[:3].lower(),
        "word.suffix2": word[-2:].lower(),
        "word.suffix3": word[-3:].lower(),
        "word.has_hyphen": int("-" in word),
        "word.has_digit": int(any(c.isdigit() for c in word)),
        "word.in_person_names": int(word in PERSON_NAMES),
        "word.is_org_suffix": int(word in ORG_SUFFIXES),
        "word.is_loc_word": int(word in LOC_WORDS),
        "word.is_punct": int(not word.isalnum() and len(word) == 1),
    }
    # 前一个词
    if i > 0:
        prev = tokens[i - 1]
        features.update({
            "prev.lower": prev.lower(),
            "prev.istitle": int(prev.istitle()),
            "prev.isupper": int(prev.isupper()),
            "prev.in_person_names": int(prev in PERSON_NAMES),
        })
    else:
        features["BOS"] = 1.0  # Beginning of Sentence

    # 后一个词
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        features.update({
            "next.lower": nxt.lower(),
            "next.istitle": int(nxt.istitle()),
            "next.isupper": int(nxt.isupper()),
            "next.is_org_suffix": int(nxt in ORG_SUFFIXES),
        })
    else:
        features["EOS"] = 1.0  # End of Sentence

    # 两步前
    if i > 1:
        pp = tokens[i - 2]
        features["pp.lower"] = pp.lower()
        features["pp.istitle"] = int(pp.istitle())

    # 两步后
    if i < len(tokens) - 2:
        nn = tokens[i + 2]
        features["nn.lower"] = nn.lower()
        features["nn.istitle"] = int(nn.istitle())

    return features


def build_feature_vocab(all_feats: list) -> dict:
    vocab = {}
    for feat_dict in all_feats:
        for k, v in feat_dict.items():
            if isinstance(v, str):
                key = f"{k}={v}"
                if key not in vocab:
                    vocab[key] = len(vocab)
            else:
                if k not in vocab:
                    vocab[k] = len(vocab)
    return vocab


def feats_to_vector(feat_dict: dict, vocab: dict) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for k, v in feat_dict.items():
        if isinstance(v, str):
            key = f"{k}={v}"
            if key in vocab:
                vec[vocab[key]] = 1.0
        else:
            if k in vocab:
                vec[vocab[k]] = float(v)
    return vec


def sentences_to_xy(sentences: list, vocab: dict = None):
    """将句子列表转换为 (X, y) 矩阵，同时返回 vocab（若未提供则新建）"""
    all_feats, all_labels = [], []
    for tokens, labels in sentences:
        for i, (tok, lab) in enumerate(zip(tokens, labels)):
            all_feats.append(word_features(tokens, i))
            all_labels.append(LABEL2ID[lab])

    if vocab is None:
        vocab = build_feature_vocab(all_feats)

    X = np.stack([feats_to_vector(f, vocab) for f in all_feats])
    y = np.array(all_labels, dtype=np.int32)
    return X, y, vocab


# ─────────────────────────── 逻辑回归（手写，梯度下降） ─────────────────────────────

class SoftmaxClassifier:
    """多分类 Logistic Regression，梯度下降 + L2 正则"""

    def __init__(self, n_classes: int, lr: float = 0.1, n_epochs: int = 30,
                 l2: float = 1e-3, batch_size: int = 64, seed: int = 42):
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.b = None
        self.losses = []

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.W = self.rng.normal(0, 0.01, (d, self.n_classes)).astype(np.float32)
        self.b = np.zeros(self.n_classes, dtype=np.float32)
        self.losses = []

        for epoch in range(self.n_epochs):
            idx = self.rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                bi = idx[start: start + self.batch_size]
                Xb, yb = X[bi], y[bi]
                logits = Xb @ self.W + self.b
                probs  = self._softmax(logits)

                # Cross-entropy loss
                batch_loss = -np.log(probs[np.arange(len(yb)), yb] + 1e-10).mean()
                epoch_loss += batch_loss * len(yb)

                # Gradient
                probs[np.arange(len(yb)), yb] -= 1.0
                probs /= len(yb)
                dW = Xb.T @ probs + self.l2 * self.W
                db = probs.sum(axis=0)

                self.W -= self.lr * dW
                self.b -= self.lr * db

            self.losses.append(epoch_loss / n)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        return self._softmax(logits).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W + self.b)


# ─────────────────────────── BIO 后处理 ─────────────────────────────

def bio_fix(labels: list) -> list:
    """修正非法 BIO 序列，如 I-X 前面没有 B-X / I-X"""
    fixed = []
    prev_entity = None
    for lab in labels:
        if lab.startswith("I-"):
            entity = lab[2:]
            if prev_entity != entity:
                lab = "B-" + entity   # 修正为 B-
        if lab == "O":
            prev_entity = None
        else:
            prev_entity = lab[2:]
        fixed.append(lab)
    return fixed


# ─────────────────────────── 评估 ─────────────────────────────

def compute_f1(true_labels: list, pred_labels: list):
    """
    计算精确率、召回率、F1（实体级别）
    使用 span-level 评估：连续实体跨度需完全匹配
    """
    def extract_spans(labels):
        spans = set()
        start, cur_type = None, None
        for i, lab in enumerate(labels):
            if lab.startswith("B-"):
                if cur_type is not None:
                    spans.add((start, i - 1, cur_type))
                start, cur_type = i, lab[2:]
            elif lab.startswith("I-"):
                etype = lab[2:]
                if cur_type != etype:
                    if cur_type is not None:
                        spans.add((start, i - 1, cur_type))
                    start, cur_type = i, etype
            else:  # O
                if cur_type is not None:
                    spans.add((start, i - 1, cur_type))
                cur_type = None
        if cur_type is not None:
            spans.add((start, len(labels) - 1, cur_type))
        return spans

    true_spans = extract_spans(true_labels)
    pred_spans = extract_spans(pred_labels)

    tp = len(true_spans & pred_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    # per-class
    per_class = {}
    for etype in ["PER", "LOC", "ORG"]:
        t = {s for s in true_spans if s[2] == etype}
        p = {s for s in pred_spans  if s[2] == etype}
        c_tp = len(t & p)
        c_fp = len(p - t)
        c_fn = len(t - p)
        pr = c_tp / (c_tp + c_fp + 1e-10)
        rc = c_tp / (c_tp + c_fn + 1e-10)
        f  = 2 * pr * rc / (pr + rc + 1e-10)
        per_class[etype] = {"precision": pr, "recall": rc, "f1": f,
                             "support": len(t)}

    return {"precision": precision, "recall": recall, "f1": f1,
            "per_class": per_class}


# ─────────────────────────── 可视化 ─────────────────────────────

ENTITY_COLORS = {"PER": "#FF6B6B", "LOC": "#4ECDC4", "ORG": "#45B7D1", "O": "#F0F0F0"}


def plot_results(classifier: SoftmaxClassifier, metrics: dict, sentences_pred: list,
                 save_path: str = "results/ner_results.png"):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#1a1a2e")

    # ── 1. 训练损失曲线 ──
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(classifier.losses, color="#FF6B6B", linewidth=2)
    ax1.set_title("Training Loss", color="white", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Cross-Entropy Loss", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    ax1.grid(True, alpha=0.2)

    # ── 2. Per-class F1 柱状图 ──
    ax2 = fig.add_subplot(3, 3, 2)
    classes = ["PER", "LOC", "ORG"]
    colors  = [ENTITY_COLORS[c] for c in classes]
    f1s     = [metrics["per_class"][c]["f1"] for c in classes]
    prs     = [metrics["per_class"][c]["precision"] for c in classes]
    rcs     = [metrics["per_class"][c]["recall"] for c in classes]

    x = np.arange(len(classes))
    w = 0.25
    ax2.bar(x - w, prs, w, label="Precision", color="#FF6B6B", alpha=0.85)
    ax2.bar(x,     f1s, w, label="F1",        color="#4ECDC4", alpha=0.85)
    ax2.bar(x + w, rcs, w, label="Recall",    color="#45B7D1", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, color="white")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Per-Class P / F1 / R", color="white", fontsize=11, fontweight="bold")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    ax2.grid(True, alpha=0.2, axis="y")

    # ── 3. Overall Metrics 雷达图 ──
    ax3 = fig.add_subplot(3, 3, 3, polar=True)
    cats   = ["Precision", "Recall", "F1"]
    vals   = [metrics["precision"], metrics["recall"], metrics["f1"]]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    vals_c = vals + [vals[0]]
    ang_c  = angles + [angles[0]]
    ax3.plot(ang_c, vals_c, color="#FF6B6B", linewidth=2)
    ax3.fill(ang_c, vals_c, color="#FF6B6B", alpha=0.3)
    ax3.set_xticks(angles)
    ax3.set_xticklabels(cats, color="white", fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.set_facecolor("#16213e")
    ax3.tick_params(colors="white")
    ax3.set_title("Overall Metrics", color="white", fontsize=11, fontweight="bold", pad=15)
    ax3.grid(color="#444", alpha=0.4)

    # ── 4. 特征权重热力图（Top 20 per class）──
    ax4 = fig.add_subplot(3, 1, 2)
    W_abs = np.abs(classifier.W)       # (n_features, n_classes)
    top_n = 15
    top_idx = np.argsort(W_abs.sum(axis=1))[-top_n:][::-1]
    W_sub   = classifier.W[top_idx].T  # (n_classes, top_n)
    im = ax4.imshow(W_sub, aspect="auto", cmap="RdYlGn",
                    vmin=-W_sub.std()*2, vmax=W_sub.std()*2)
    label_names = [ID2LABEL[i] for i in range(N_LABELS)]
    ax4.set_yticks(range(N_LABELS))
    ax4.set_yticklabels(label_names, color="white", fontsize=9)
    ax4.set_xticks(range(top_n))
    ax4.set_xticklabels([f"feat_{i}" for i in top_idx], rotation=45, ha="right",
                         color="white", fontsize=7)
    ax4.set_title("Feature Weight Heatmap (Top 15 Features)", color="white",
                  fontsize=11, fontweight="bold")
    ax4.set_facecolor("#16213e")
    plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.02)

    # ── 5. 句子实体标注可视化 ──
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis("off")
    ax5.set_facecolor("#16213e")
    ax5.set_title("NER Annotation Examples (Test Set)", color="white",
                  fontsize=11, fontweight="bold")

    n_sent = min(3, len(sentences_pred))
    row_h  = 0.28
    for si, (tokens, true_labels, pred_labels) in enumerate(sentences_pred[:n_sent]):
        y_top = 0.95 - si * row_h
        x_cur = 0.01
        for tok, tl, pl in zip(tokens, true_labels, pred_labels):
            etype = pl[2:] if pl != "O" else "O"
            color = ENTITY_COLORS.get(etype, "#F0F0F0")
            bg = "#1a1a2e" if etype == "O" else color
            fc = "white"   if etype == "O" else "black"

            bbox_props = dict(boxstyle="round,pad=0.15", facecolor=bg,
                              edgecolor=color, linewidth=1.2)
            t = ax5.text(x_cur, y_top, tok, color=fc, fontsize=9,
                         va="center", bbox=bbox_props)
            x_cur += 0.015 * (len(tok) + 1.5)
            if x_cur > 0.95:
                x_cur = 0.01
                y_top -= 0.07

    legend_patches = [mpatches.Patch(color=ENTITY_COLORS[c], label=c)
                      for c in ["PER", "LOC", "ORG", "O"]]
    ax5.legend(handles=legend_patches, loc="lower right",
               facecolor="#0f3460", labelcolor="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Named Entity Recognition — BIO Sequence Labeling", color="white",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ─────────────────────────── 主函数 ─────────────────────────────

def named_entity_recognition():
    """命名实体识别完整流程"""
    print("=" * 60)
    print("   命名实体识别 (Named Entity Recognition, NER)")
    print("=" * 60)

    # ── 1. 构建数据 ──
    print("\n[1/5] 准备数据集...")
    X_train, y_train, vocab = sentences_to_xy(TRAIN_SENTENCES)
    X_test,  y_test,  _     = sentences_to_xy(TEST_SENTENCES, vocab)
    print(f"  训练集: {len(X_train)} 词, 特征维度: {X_train.shape[1]}")
    print(f"  测试集: {len(X_test)} 词")

    # 标签分布
    unique, counts = np.unique(y_train, return_counts=True)
    print("  训练集标签分布:")
    for uid, cnt in zip(unique, counts):
        print(f"    {ID2LABEL[uid]:<8}: {cnt}")

    # ── 2. 训练分类器 ──
    print("\n[2/5] 训练 Softmax 分类器...")
    clf = SoftmaxClassifier(n_classes=N_LABELS, lr=0.5, n_epochs=60,
                             l2=1e-3, batch_size=32, seed=42)
    clf.fit(X_train, y_train)
    print(f"  最终训练 Loss: {clf.losses[-1]:.4f}")

    # ── 3. 预测 + BIO 后处理 ──
    print("\n[3/5] 预测 & 后处理...")
    y_pred_raw = clf.predict(X_test)
    # 重组为句子序列再做 BIO 修正
    pred_label_seqs, true_label_seqs = [], []
    idx = 0
    sentences_pred = []
    for tokens, labels in TEST_SENTENCES:
        n = len(tokens)
        raw_seq = [ID2LABEL[y_pred_raw[idx + i]] for i in range(n)]
        fix_seq = bio_fix(raw_seq)
        pred_label_seqs.extend(fix_seq)
        true_label_seqs.extend(labels)
        sentences_pred.append((tokens, labels, fix_seq))
        idx += n

    # ── 4. 评估 ──
    print("\n[4/5] 评估结果...")
    metrics = compute_f1(true_label_seqs, pred_label_seqs)
    print(f"  Overall Precision : {metrics['precision']:.4f}")
    print(f"  Overall Recall    : {metrics['recall']:.4f}")
    print(f"  Overall F1        : {metrics['f1']:.4f}")
    print()
    print(f"  {'Entity':<8} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}")
    print("  " + "-" * 42)
    for etype in ["PER", "LOC", "ORG"]:
        pc = metrics["per_class"][etype]
        print(f"  {etype:<8} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
              f"{pc['f1']:>8.4f} {pc['support']:>8}")

    # ── 5. 展示预测示例 ──
    print("\n[5/5] 预测示例:")
    for tokens, true_labels, pred_labels in sentences_pred:
        print(f"\n  句子: {' '.join(tokens)}")
        print(f"  真实: {' '.join(true_labels)}")
        print(f"  预测: {' '.join(pred_labels)}")

    # ── 可视化 ──
    print("\n生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(clf, metrics, sentences_pred, save_path="results/ner_results.png")

    print("\n[DONE] 命名实体识别完成!")
    return clf, metrics


if __name__ == "__main__":
    named_entity_recognition()
