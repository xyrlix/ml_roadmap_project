# 多项式朴素贝叶斯 (Multinomial Naive Bayes)
# 适用于离散计数特征（如词频），P(x_i|y) = Multinomial(θ_y)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils import get_results_path, save_and_close


# ───────────────────────────── 合成文本数据 ─────────────────────────────

# 简单玩具文本分类数据集（2类：科技/体育）
DOCUMENTS = [
    ("AI model uses deep learning neural networks", "tech"),
    ("Machine learning trains on large datasets", "tech"),
    ("Python is popular for data science", "tech"),
    ("Neural networks learn patterns from data", "tech"),
    ("Natural language processing analyzes text", "tech"),
    ("TensorFlow framework builds deep models", "tech"),
    ("Computer vision recognizes images objects", "tech"),
    ("Gradient descent optimizes model weights", "tech"),
    ("Backpropagation trains neural networks", "tech"),
    ("Artificial intelligence transforms industries", "tech"),
    ("Football team scored winning goal", "sports"),
    ("Basketball player made three point shot", "sports"),
    ("Tennis match lasted three hours", "sports"),
    ("Soccer goalkeeper saved penalty kick", "sports"),
    ("Baseball hitter hit home run", "sports"),
    ("Swimming competition set new record", "sports"),
    ("Athletics runner broke world record", "sports"),
    ("Golf champion won major tournament", "sports"),
    ("Hockey team defended the puck", "sports"),
    ("Racing car completed the lap", "sports"),
]

np.random.shuffle(DOCUMENTS)


# ───────────────────────────── 手写多项式 NB ─────────────────────────────

class MultinomialNBFromScratch:
    """从零实现多项式朴素贝叶斯：用于文本分类"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha           # Laplace 平滑
        self.class_priors_ = None   # P(y)
        self.feature_probs_ = None   # P(x_i=1|y)  (词频)
        self.classes_ = None

    def fit(self, X, y):
        """X 是文档-词矩阵（词频计数）"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_priors_ = np.zeros(n_classes)
        self.feature_probs_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            # P(y) = (|D_c| + α) / (|D| + α * n_classes)
            self.class_priors_[i] = (len(X_c) + self.alpha) / (len(X) + self.alpha * n_classes)

            # P(w|y) = (count(w,c) + α) / (Σ_w count(w,c) + α * V)
            word_counts = X_c.sum(axis=0) + self.alpha
            total_words = word_counts.sum()
            self.feature_probs_[i] = word_counts / (total_words + 1e-10)

        return self

    def predict_log_proba(self, X):
        """计算 log P(y|x)"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # log P(y) + Σ x_i * log P(w_i|y)
            log_y = np.log(self.class_priors_[i] + 1e-10)
            log_likelihood = (X @ np.log(self.feature_probs_[i] + 1e-10))
            log_probs[:, i] = log_y + log_likelihood

        return log_probs

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return self.classes_[log_probs.argmax(axis=1)]

    def get_top_words(self, vectorizer, n_top=5):
        """返回每类最可能的词"""
        word_to_idx = vectorizer.vocabulary_
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        results = {}
        for i, c in enumerate(self.classes_):
            top_idx = np.argsort(self.feature_probs_[i])[-n_top:][::-1]
            top_words = [idx_to_word[idx] for idx in top_idx]
            results[c] = list(zip(top_words, self.feature_probs_[i][top_idx]))
        return results


def multinomial_nb():
    """多项式朴素贝叶斯（文本分类）实现"""
    print("多项式朴素贝叶斯 (Multinomial NB) 运行中...\n")

    # 1. 数据准备
    print("1. 准备文本数据集（科技 vs 体育）...")
    texts = [doc for doc, _ in DOCUMENTS]
    labels = [1 if cat == "tech" else 0 for _, cat in DOCUMENTS]

    # 词袋模型
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    X = vectorizer.fit_transform(texts).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels)
    print(f"   词汇量: {len(vectorizer.vocabulary_)}")
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}")

    # 2. 手写模型训练
    print("2. 训练手写多项式 NB...")
    model_scratch = MultinomialNBFromScratch(alpha=1.0)
    model_scratch.fit(X_train, y_train)
    y_pred_scratch = model_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    print(f"   手写模型准确率: {acc_scratch:.4f}")

    # 3. sklearn 对比
    print("3. sklearn MultinomialNB 对比...")
    model_sk = MultinomialNB(alpha=1.0)
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"   sklearn 准确率: {acc_sk:.4f}")

    # 4. 每类 Top 词
    print("4. 提取每类最可能的词...")
    top_words = model_scratch.get_top_words(vectorizer, n_top=6)
    for c, word_probs in top_words.items():
        label = "Tech" if c == 1 else "Sports"
        print(f"   {label}:")
        for word, prob in word_probs:
            print(f"     {word:15s}  P(w|y={label})={prob:.4f}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：准确率对比 ──
    ax = axes[0]
    models = ['Scratch', 'sklearn']
    accs = [acc_scratch, acc_sk]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(models, accs, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('模型准确率对比')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图2：Tech 类词概率 Top 10 ──
    ax = axes[1]
    tech_probs = top_words[1]
    words, probs = zip(*tech_probs)
    bars = ax.barh(range(len(words)), probs, color='#e74c3c', alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_title('Tech 类最可能词（Top 6）')
    ax.set_xlabel('P(word | Tech)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # ── 子图3：Sports 类词概率 Top 10 ──
    ax = axes[2]
    sports_probs = top_words[0]
    words, probs = zip(*sports_probs)
    bars = ax.barh(range(len(words)), probs, color='#3498db', alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_title('Sports 类最可能词（Top 6）')
    ax.set_xlabel('P(word | Sports)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    save_path = get_results_path('naive_bayes_multinomial_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    multinomial_nb()
