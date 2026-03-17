# 贝叶斯网络模型
# 有向无环图（DAG）表示变量间的条件依赖关系，使用条件概率推断

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from utils import get_results_path, save_and_close


# ──────────────────────────────────────────────────────────────────────
# 手动实现简单贝叶斯网络（离散变量，条件概率表）
# 问题：疾病诊断
#   D (Disease) -> S (Symptom1), F (Fever)
#   S -> T (Test positive)
# ──────────────────────────────────────────────────────────────────────

class BayesianNetwork:
    """
    简单贝叶斯网络：
    P(D) = 先验
    P(S|D), P(F|D) = 似然
    P(T|S) = 条件概率
    """
    def __init__(self):
        # 先验 P(Disease=1) = 0.1
        self.p_d = {'0': 0.9, '1': 0.1}

        # P(Symptom | Disease)
        self.p_s_given_d = {
            ('1', '1'): 0.80,   # P(S=1|D=1)
            ('1', '0'): 0.20,   # P(S=1|D=0)
        }

        # P(Fever | Disease)
        self.p_f_given_d = {
            ('1', '1'): 0.70,
            ('1', '0'): 0.10,
        }

        # P(TestPos | Symptom)
        self.p_t_given_s = {
            ('1', '1'): 0.90,   # P(T=1|S=1)
            ('1', '0'): 0.15,   # P(T=1|S=0)
        }

    def _p_s(self, s, d):
        key = (str(s), str(d))
        p = self.p_s_given_d.get(key, None)
        if p is None:
            return 1 - self.p_s_given_d.get(('1', str(d)), 0.5)
        return p

    def _p_f(self, f, d):
        key = (str(f), str(d))
        p = self.p_f_given_d.get(key, None)
        if p is None:
            return 1 - self.p_f_given_d.get(('1', str(d)), 0.5)
        return p

    def _p_t(self, t, s):
        key = (str(t), str(s))
        p = self.p_t_given_s.get(key, None)
        if p is None:
            return 1 - self.p_t_given_s.get(('1', str(s)), 0.5)
        return p

    def posterior(self, symptom, fever, test_pos):
        """P(Disease=1 | S, F, T) via Bayes' theorem"""
        joint_d1 = (self.p_d['1']
                    * self._p_s(symptom, 1)
                    * self._p_f(fever, 1)
                    * self._p_t(test_pos, symptom))
        joint_d0 = (self.p_d['0']
                    * self._p_s(symptom, 0)
                    * self._p_f(fever, 0)
                    * self._p_t(test_pos, symptom))
        total = joint_d1 + joint_d0
        return joint_d1 / total if total > 0 else 0


def generate_medical_data(n=1000, seed=42):
    """生成模拟医疗数据"""
    rng = np.random.default_rng(seed)
    bn = BayesianNetwork()
    rows = []
    for _ in range(n):
        d = rng.binomial(1, 0.1)
        s = rng.binomial(1, 0.8 if d == 1 else 0.2)
        f = rng.binomial(1, 0.7 if d == 1 else 0.1)
        t = rng.binomial(1, 0.9 if s == 1 else 0.15)
        rows.append([d, s, f, t])
    return np.array(rows)


def bayesian_network():
    """贝叶斯网络概率推断实现"""
    print("贝叶斯网络模型运行中...\n")

    # 1. 生成数据 & 手动 BN 推断
    print("1. 构建贝叶斯网络，演示后验概率推断...")
    bn = BayesianNetwork()
    test_cases = [
        (0, 0, 0, "无症状、无发烧、阴性"),
        (1, 0, 0, "有症状、无发烧、阴性"),
        (1, 1, 0, "有症状、有发烧、阴性"),
        (1, 1, 1, "有症状、有发烧、阳性"),
        (0, 1, 1, "无症状、有发烧、阳性"),
    ]
    print("   患病后验概率（P(Disease=1 | 观测)）：")
    case_labels, posteriors = [], []
    for s, f, t, desc in test_cases:
        p = bn.posterior(s, f, t)
        print(f"   {desc}: {p:.4f}")
        case_labels.append(desc[:8])
        posteriors.append(p)

    # 2. 用 Naive Bayes 分类器验证（sklearn）
    print("\n2. 高斯朴素贝叶斯（Naive Bayes）分类器验证...")
    data = generate_medical_data(n=2000)
    X = data[:, 1:]   # [S, F, T]
    y = data[:, 0]    # D

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   GaussianNB 准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Diseased']))

    # 3. 先验概率敏感性分析
    print("3. 先验 P(D) 对后验的影响...")
    prior_range = np.linspace(0.01, 0.5, 20)
    post_vals = []
    for pr in prior_range:
        bn_tmp = BayesianNetwork()
        bn_tmp.p_d = {'1': pr, '0': 1 - pr}
        post_vals.append(bn_tmp.posterior(1, 1, 1))

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：不同观测下的后验概率 ──
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(np.array(posteriors))
    bars = ax.barh(range(len(case_labels)), posteriors,
                   color=colors, edgecolor='white')
    for bar, p in zip(bars, posteriors):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{p:.3f}', va='center', fontsize=10)
    ax.set_yticks(range(len(case_labels)))
    ax.set_yticklabels(case_labels, fontsize=8)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('P(Disease=1 | Evidence)')
    ax.set_title('贝叶斯后验概率（不同证据组合）')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')

    # ── 子图2：贝叶斯网络结构图 ──
    ax = axes[1]
    ax.axis('off')
    nodes = {'D': (0.5, 0.85), 'S': (0.2, 0.5),
             'F': (0.8, 0.5), 'T': (0.2, 0.15)}
    node_labels = {
        'D': 'Disease\nP(D)=0.1',
        'S': 'Symptom\nP(S|D)',
        'F': 'Fever\nP(F|D)',
        'T': 'Test\nP(T|S)'
    }
    edges = [('D', 'S'), ('D', 'F'), ('S', 'T')]
    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.08, color='steelblue', alpha=0.7, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, node_labels[name], ha='center', va='center',
                fontsize=8, zorder=4, color='white', fontweight='bold')
    for src, dst in edges:
        xs, ys = nodes[src]
        xd, yd = nodes[dst]
        ax.annotate('', xy=(xd, yd), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('贝叶斯网络结构（DAG）')

    # ── 子图3：先验 P(D) 对后验的影响 ──
    ax = axes[2]
    ax.plot(prior_range, post_vals, 'bo-', linewidth=1.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='阈值=0.5')
    ax.set_xlabel('先验 P(Disease=1)')
    ax.set_ylabel('后验 P(D=1 | S=1, F=1, T=1)')
    ax.set_title('先验概率 vs 后验概率')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('bayesian_network_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    bayesian_network()
