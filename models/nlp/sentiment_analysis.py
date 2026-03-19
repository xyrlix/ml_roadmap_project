# 情感分析模型
# 使用 TF-IDF + 机器学习 & 简单神经网络实现文本情感分类

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from utils import get_results_path, save_and_close


# ─────────────────────────────────────────────────────────────
# 合成情感数据集（含正面/负面/中性三类）
# ─────────────────────────────────────────────────────────────
POSITIVE_TEXTS = [
    "This product is absolutely amazing and I love it",
    "Excellent quality and fast delivery highly recommend",
    "The best purchase I have ever made wonderful",
    "Very happy with this great product and great service",
    "Outstanding performance and beautiful design love it",
    "Fantastic experience the team was very helpful",
    "Superb quality exceeded my expectations brilliant",
    "Really impressed with the quality and the price",
    "Wonderful product works perfectly and looks great",
    "Awesome value for money totally satisfied",
    "Great service friendly staff very professional",
    "Love this product it changed my life completely",
    "Perfect gift my family absolutely loved it",
    "Highly recommend this to everyone amazing quality",
    "Best experience ever customer support was excellent",
    "This is fantastic I am thrilled with my purchase",
    "Incredible product superior quality beyond expectations",
    "So happy with this bought two more for friends",
    "Amazing results very pleased with everything",
    "Top quality service delivered fast no complaints",
    "Five stars this is exactly what I needed",
    "Brilliant product would definitely buy again",
    "Very satisfied customer will come back",
    "Exceeded all my expectations simply outstanding",
    "Wonderful experience from start to finish",
]

NEGATIVE_TEXTS = [
    "This product is terrible and I hate it",
    "Very poor quality broken after one day awful",
    "Worst purchase ever complete waste of money",
    "Very disappointed with this product and service",
    "Horrible experience customer service was useless",
    "Total garbage returned immediately disgusting quality",
    "Never buying from here again absolute rubbish",
    "Failed to work immediately disappointed and frustrated",
    "Terrible quality fell apart after one use",
    "Complete waste complete scam avoid at all costs",
    "Rude staff awful service will not return",
    "This is defective and the company ignored my complaint",
    "Overpriced junk arrived damaged terrible packaging",
    "Worst product I have ever owned total disaster",
    "Completely useless does not work as advertised",
    "Very bad experience would not recommend to anyone",
    "Frustrating waste of time and money very disappointed",
    "Poor quality fails to deliver on promises",
    "Terrible delivery arrived late and damaged horrible",
    "Awful product cheap materials broke immediately",
    "Zero stars if I could very bad very disappointed",
    "Horrible product smells bad looks worse useless",
    "Deeply unhappy with this order complete mess",
    "The worst experience of my life avoid completely",
    "Not working at all terrible customer care",
]

NEUTRAL_TEXTS = [
    "The product is okay nothing special",
    "Average quality for the price acceptable",
    "It does what it says nothing more nothing less",
    "Reasonable product moderate quality fair price",
    "Works as expected standard product",
    "Neither good nor bad just average",
    "It is fine I suppose not amazing not terrible",
    "The product arrived on time and works",
    "Acceptable performance standard quality",
    "It is a normal product does the job",
    "Okay product meets basic requirements",
    "Delivered on time product is as described",
    "Standard purchase nothing to complain about",
    "It does its job adequately average product",
    "Ordinary product ordinary experience",
    "About what I expected neither impressed nor disappointed",
    "The item is functional but unremarkable",
    "Decent quality for the price no issues",
    "Works as described not bad not great",
    "A solid if unremarkable product does what it says",
    "Average product average service",
    "Not spectacular but gets the job done",
    "Meets expectations no complaints",
    "Ordinary and adequate typical purchase",
    "Fine product works normally",
]


def build_dataset():
    texts = POSITIVE_TEXTS + NEGATIVE_TEXTS + NEUTRAL_TEXTS
    labels = ([2] * len(POSITIVE_TEXTS) +
              [0] * len(NEGATIVE_TEXTS) +
              [1] * len(NEUTRAL_TEXTS))
    return texts, np.array(labels)


# ─────────────────────────────────────────────────────────────
# 简单前馈神经网络（numpy）用于情感分类
# ─────────────────────────────────────────────────────────────
class SimpleTextNN:
    """单隐层神经网络，输入为 TF-IDF 特征向量"""
    def __init__(self, in_dim, hidden_dim=64, n_classes=3,
                 lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (in_dim + hidden_dim))
        self.W1 = rng.normal(0, std, (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        std2 = np.sqrt(2.0 / (hidden_dim + n_classes))
        self.W2 = rng.normal(0, std2, (hidden_dim, n_classes))
        self.b2 = np.zeros(n_classes)
        self.lr = lr

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / (e.sum(axis=1, keepdims=True) + 1e-10)

    def forward(self, X):
        self._X = X
        self._H = self._relu(X @ self.W1 + self.b1)
        logits = self._H @ self.W2 + self.b2
        return self._softmax(logits)

    def train_step(self, X, y_onehot):
        probs = self.forward(X)
        loss = -np.mean((np.log(np.clip(probs, 1e-10, 1)) * y_onehot).sum(axis=1))

        # 反向
        delta2 = (probs - y_onehot) / len(X)
        dW2 = self._H.T @ delta2
        db2 = delta2.sum(axis=0)
        dH = delta2 @ self.W2.T * (self._H > 0)
        dW1 = X.T @ dH
        db1 = dH.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return loss

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def sentiment_analysis():
    """情感分析实现（TF-IDF + 多模型对比）"""
    print("情感分析模型运行中...\n")

    # 1. 数据准备
    print("1. 准备情感分析数据集...")
    texts, labels = build_dataset()
    label_names = ['负面', '中性', '正面']
    print(f"   总样本数: {len(texts)}  类别分布: "
          f"负面={sum(labels==0)} 中性={sum(labels==1)} 正面={sum(labels==2)}")

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels)

    # 2. TF-IDF 特征提取
    print("2. TF-IDF 特征提取...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500,
                            sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train_txt).toarray()
    X_test  = tfidf.transform(X_test_txt).toarray()
    print(f"   特征维度: {X_train.shape[1]}")

    # 3. 多模型对比
    print("3. 训练并对比多个分类器...")
    sklearn_models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Linear SVC':          LinearSVC(max_iter=500, random_state=42),
        'Naive Bayes':         MultinomialNB(alpha=0.1),
    }

    results = {}
    cms = {}
    for name, clf in sklearn_models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {'acc': acc, 'cv_mean': cv_scores.mean(),
                         'cv_std': cv_scores.std()}
        cms[name] = confusion_matrix(y_test, y_pred)
        print(f"   {name:22s}  Test Acc={acc:.4f}  "
              f"CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    # 4. 神经网络
    print("4. 训练简单神经网络...")
    y_train_oh = np.zeros((len(y_train), 3))
    for i, c in enumerate(y_train):
        y_train_oh[i, c] = 1.0

    nn = SimpleTextNN(X_train.shape[1], hidden_dim=64, n_classes=3,
                      lr=0.05, seed=42)
    nn_losses = []
    EPOCHS = 200
    batch = len(X_train)
    for ep in range(EPOCHS):
        loss = nn.train_step(X_train, y_train_oh)
        nn_losses.append(loss)

    nn_pred = nn.predict(X_test)
    nn_acc = accuracy_score(y_test, nn_pred)
    results['Neural Network'] = {'acc': nn_acc, 'cv_mean': None, 'cv_std': None}
    cms['Neural Network'] = confusion_matrix(y_test, nn_pred)
    print(f"   {'Neural Network':22s}  Test Acc={nn_acc:.4f}")

    # 5. 测试样本预测示例
    print("\n5. 示例预测...")
    best_model_name = max(sklearn_models, key=lambda k: results[k]['acc'])
    best_model = sklearn_models[best_model_name]
    demo_texts = [
        "This is absolutely wonderful I love it",
        "Terrible product broke immediately waste of money",
        "It works fine nothing special about it",
    ]
    demo_feats = tfidf.transform(demo_texts).toarray()
    demo_preds = best_model.predict(demo_feats)
    for txt, pred in zip(demo_texts, demo_preds):
        print(f"   '{txt[:40]}...' → {label_names[pred]}")

    # 6. 可视化
    print("\n6. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：各模型准确率对比 ──
    ax = axes[0]
    model_names = list(results.keys())
    accs = [results[n]['acc'] for n in model_names]
    cv_means = [results[n]['cv_mean'] or 0 for n in model_names]
    cv_stds  = [results[n]['cv_std']  or 0 for n in model_names]
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, accs, color='steelblue', edgecolor='white',
                  alpha=0.85, width=0.5)
    ax.errorbar(x_pos[:3], cv_means[:3], yerr=cv_stds[:3],
                fmt='none', color='darkorange', capsize=5, lw=2,
                label='CV Mean±Std')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
    ax.set_title('情感分析：各模型准确率')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图2：最佳模型混淆矩阵 ──
    ax = axes[1]
    cm = cms[best_model_name]
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=12)
    ax.set_title(f'混淆矩阵 — {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # ── 子图3：神经网络损失曲线 + TF-IDF 词云（重要词汇） ──
    ax = axes[2]
    ax.plot(nn_losses, color='steelblue', lw=1.5)
    ax.set_title('神经网络训练损失')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95,
            f'Final Acc={nn_acc:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('情感分析 — TF-IDF + 多分类器对比', fontsize=13, y=1.01)
    save_path = get_results_path('sentiment_analysis_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    sentiment_analysis()
