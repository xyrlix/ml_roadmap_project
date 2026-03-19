"""
特征编码方法 (Feature Encoding)
=================================
实现并对比处理分类特征、数值特征的常用编码策略

实现方法：
  类别特征编码：
    1. One-Hot Encoding（独热编码）
    2. Label Encoding（标签编码）
    3. Target Encoding（目标均值编码，带平滑）
    4. Frequency Encoding（频率编码）
    5. Binary Encoding（二进制编码，高基数特征）
    6. Ordinal Encoding（有序编码）
  数值特征变换：
    7. StandardScaler / MinMaxScaler / RobustScaler
    8. PowerTransformer (Box-Cox / Yeo-Johnson)
    9. Quantile Transformer（分位数归一化）
    10. 分箱 (Binning / Discretization)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 目标编码（手写） ────────────────────────

class TargetEncoder:
    """
    Target Encoding（均值目标编码 + 贝叶斯平滑）
    encoded = (count * category_mean + k * global_mean) / (count + k)
    k: 平滑强度，防止低频类别过拟合
    """
    def __init__(self, k=10):
        self.k = k
        self.global_mean_ = None
        self.mapping_ = {}

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        unique_vals = np.unique(X)
        for v in unique_vals:
            mask = (X == v)
            count = mask.sum()
            cat_mean = y[mask].mean()
            # 贝叶斯平滑
            self.mapping_[v] = (count * cat_mean + self.k * self.global_mean_) / (count + self.k)
        return self

    def transform(self, X):
        return np.array([self.mapping_.get(v, self.global_mean_) for v in X])

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class FrequencyEncoder:
    """频率编码：用类别在训练集中出现的频率替换类别值"""
    def __init__(self):
        self.mapping_ = {}

    def fit(self, X):
        unique, counts = np.unique(X, return_counts=True)
        total = len(X)
        self.mapping_ = {v: c / total for v, c in zip(unique, counts)}
        return self

    def transform(self, X):
        return np.array([self.mapping_.get(v, 0.0) for v in X])

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def feature_encoding():
    print("特征编码方法运行中...\n")

    np.random.seed(42)

    # ── 生成模拟数据集 ────────────────────────────────────────────
    print("1. 生成含类别特征的数据集...")
    N = 1000

    # 模拟一个电商用户转化预测场景
    # 类别特征
    cities     = np.random.choice(["Beijing", "Shanghai", "Guangzhou", "Shenzhen",
                                    "Hangzhou", "Chengdu", "Wuhan", "Nanjing"], N,
                                   p=[0.25, 0.22, 0.15, 0.12, 0.1, 0.07, 0.05, 0.04])
    devices    = np.random.choice(["mobile", "desktop", "tablet"], N, p=[0.6, 0.3, 0.1])
    age_group  = np.random.choice(["18-24", "25-34", "35-44", "45+"], N, p=[0.25, 0.35, 0.25, 0.15])

    # 数值特征（含偏态分布）
    age      = np.random.gamma(3, 8, N) + 18         # 右偏
    income   = np.random.lognormal(10, 0.8, N)       # 对数正态（收入）
    pageview = np.random.exponential(5, N)            # 指数分布
    bounce_rate = np.random.beta(5, 2, N)             # Beta 分布

    # 目标变量（转化=1）
    logit = (
        0.5 * (cities == "Shanghai").astype(float)
        + 0.3 * (devices == "mobile").astype(float)
        - 0.2 * (age_group == "45+").astype(float)
        + 0.001 * income
        + 0.05 * pageview
        - 0.3 * bounce_rate
        + np.random.normal(0, 0.5, N)
    )
    y = (logit > np.percentile(logit, 50)).astype(int)

    # 处理类别特征
    print("2. 类别特征编码对比...")

    # LabelEncoder
    le_city = LabelEncoder()
    city_label = le_city.fit_transform(cities)

    # OneHot
    ohe = OneHotEncoder(sparse_output=False)
    city_onehot = ohe.fit_transform(cities.reshape(-1, 1))

    # Target Encoding
    te = TargetEncoder(k=10)
    city_target = te.fit_transform(cities, y)

    # Frequency Encoding
    fe = FrequencyEncoder()
    city_freq = fe.fit_transform(cities)

    # Binary Encoding（手工实现）
    label_int = le_city.transform(cities)
    n_bits = int(np.ceil(np.log2(len(le_city.classes_) + 1)))
    city_binary = np.array([
        [int(b) for b in format(v, f"0{n_bits}b")]
        for v in label_int
    ])

    # 数值特征变换
    print("3. 数值特征分布变换...")
    income_orig = income.copy()
    ss = StandardScaler()
    mm = MinMaxScaler()
    rb = RobustScaler()
    pt_bc = PowerTransformer(method="yeo-johnson")
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")

    income_ss  = ss.fit_transform(income.reshape(-1, 1)).ravel()
    income_mm  = mm.fit_transform(income.reshape(-1, 1)).ravel()
    income_rb  = rb.fit_transform(income.reshape(-1, 1)).ravel()
    income_pt  = pt_bc.fit_transform(income.reshape(-1, 1)).ravel()
    income_qt  = qt.fit_transform(income.reshape(-1, 1)).ravel()
    income_bin = kbd.fit_transform(income.reshape(-1, 1)).ravel()

    # 评估各类别编码方法的效果
    print("4. 评估不同编码方法的分类性能...")
    dev_le  = LabelEncoder().fit_transform(devices)
    dev_ohe = OneHotEncoder(sparse_output=False).fit_transform(devices.reshape(-1, 1))
    age_ord = OrdinalEncoder(categories=[["18-24", "25-34", "35-44", "45+"]]).fit_transform(
        age_group.reshape(-1, 1)).ravel()

    def make_feature_set(city_enc):
        """构建特征矩阵"""
        if city_enc.ndim == 1:
            city_enc = city_enc.reshape(-1, 1)
        return np.hstack([
            city_enc,
            dev_le.reshape(-1, 1),
            age_ord.reshape(-1, 1),
            age.reshape(-1, 1),
            income.reshape(-1, 1),
            pageview.reshape(-1, 1),
        ])

    encodings = {
        "Label Enc":    make_feature_set(city_label.astype(float)),
        "One-Hot":      np.hstack([city_onehot, dev_le.reshape(-1,1), age_ord.reshape(-1,1),
                                   age.reshape(-1,1), income.reshape(-1,1), pageview.reshape(-1,1)]),
        "Target Enc":   make_feature_set(city_target),
        "Freq Enc":     make_feature_set(city_freq),
        "Binary Enc":   np.hstack([city_binary, dev_le.reshape(-1,1), age_ord.reshape(-1,1),
                                   age.reshape(-1,1), income.reshape(-1,1), pageview.reshape(-1,1)]),
    }

    scaler_eval = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    acc_results = {}
    for name, X_enc in encodings.items():
        X_s = scaler_eval.fit_transform(X_enc)
        cv = cross_val_score(clf, X_s, y, cv=5, scoring="accuracy")
        acc_results[name] = cv.mean()
        print(f"   {name:<14}: CV Acc = {cv.mean():.4f} ± {cv.std():.4f}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("5. 生成可视化...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：原始收入分布 ──────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(income_orig, bins=60, color="#3498db", alpha=0.8, edgecolor="#333")
    ax.set_title("原始收入分布（右偏对数正态）", color="white", pad=8)
    ax.set_xlabel("Income", color="#aaa"); ax.set_ylabel("Count", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图2：各种数值变换后分布 ────────────────────────────────
    transforms = {
        "StandardScaler": income_ss, "MinMaxScaler": income_mm,
        "RobustScaler": income_rb, "PowerTransform\n(YeoJohnson)": income_pt,
        "QuantileTransform\n(Normal)": income_qt,
    }
    colors_t = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    ax = axes[0, 1]
    for (name, data), color in zip(transforms.items(), colors_t):
        ax.hist(data, bins=50, alpha=0.5, color=color, label=name, density=True)
    ax.set_title("数值特征变换后分布对比", color="white", pad=8)
    ax.set_xlabel("Transformed Value", color="#aaa"); ax.set_ylabel("Density", color="#aaa")
    ax.tick_params(colors="gray")
    ax.legend(fontsize=6, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图3：分箱示例 ──────────────────────────────────────────
    ax = axes[0, 2]
    edges = np.percentile(income_orig, np.linspace(0, 100, 11))
    ax.hist(income_orig, bins=80, color="#555", alpha=0.6, label="原始")
    for e in edges[1:-1]:
        ax.axvline(e, color="#e74c3c", alpha=0.7, linewidth=1.2)
    ax.set_title("分箱 (Quantile Binning, 10 bins)", color="white", pad=8)
    ax.set_xlabel("Income", color="#aaa"); ax.set_ylabel("Count", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：城市分布（Target Encoding 平均值） ────────────────
    ax = axes[1, 0]
    city_target_mean = {c: te.mapping_.get(c, te.global_mean_) for c in le_city.classes_}
    sorted_cities = sorted(city_target_mean.keys(), key=lambda x: city_target_mean[x], reverse=True)
    vals_city = [city_target_mean[c] for c in sorted_cities]
    ax.bar(sorted_cities, vals_city,
           color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_cities))), alpha=0.85)
    ax.axhline(te.global_mean_, color="white", linestyle="--", alpha=0.7, label=f"Global Mean={te.global_mean_:.3f}")
    ax.set_title("Target Encoding（城市 vs 转化率）", color="white", pad=8)
    ax.set_ylabel("Smoothed Target Mean", color="#aaa")
    ax.tick_params(colors="gray", axis="x", rotation=30)
    ax.legend(fontsize=7, facecolor="#0d0d1a", labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图5：频率编码分布 ──────────────────────────────────────
    ax = axes[1, 1]
    city_freq_map = fe.mapping_
    sorted_cf = sorted(city_freq_map.keys(), key=lambda x: city_freq_map[x], reverse=True)
    ax.bar(sorted_cf, [city_freq_map[c] for c in sorted_cf],
           color="#3498db", alpha=0.85)
    ax.set_title("Frequency Encoding（城市出现频率）", color="white", pad=8)
    ax.set_ylabel("Frequency", color="#aaa")
    ax.tick_params(colors="gray", axis="x", rotation=30)
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图6：二进制编码展示 ────────────────────────────────────
    ax = axes[1, 2]
    cities_sample = le_city.classes_
    binary_mat = np.array([
        [int(b) for b in format(i, f"0{n_bits}b")]
        for i in range(len(cities_sample))
    ])
    im = ax.imshow(binary_mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(cities_sample)))
    ax.set_yticklabels(cities_sample, color="white", fontsize=8)
    ax.set_xticks(range(n_bits))
    ax.set_xticklabels([f"b{i}" for i in range(n_bits)], color="white")
    ax.set_title(f"Binary Encoding ({n_bits}位二进制)", color="white", pad=8)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for i in range(len(cities_sample)):
        for j in range(n_bits):
            ax.text(j, i, str(binary_mat[i, j]), ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")

    # ── 子图7：编码方法分类性能对比 ─────────────────────────────
    ax = axes[2, 0]
    names_a = list(acc_results.keys())
    vals_a  = [acc_results[n] for n in names_a]
    pal = plt.cm.plasma(np.linspace(0.3, 0.9, len(names_a)))
    bars = ax.bar(names_a, vals_a, color=pal, alpha=0.85)
    ax.set_title("各编码方法 5-Fold CV 准确率", color="white", pad=8)
    ax.set_ylabel("Accuracy", color="#aaa")
    ax.set_ylim(0.4, 0.75)
    ax.tick_params(colors="gray", axis="x", rotation=20)
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, vals_a):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=9)

    # ── 子图8：编码方法总结表 ────────────────────────────────────
    ax = axes[2, 1]
    ax.axis("off")
    table_data = [
        ["编码方法", "适用场景", "高基数", "有序"],
        ["One-Hot", "低/中基数", "否", "否"],
        ["Label Enc", "树模型", "是", "否"],
        ["Target Enc", "高基数/回归", "是", "否"],
        ["Freq Enc", "高基数", "是", "否"],
        ["Binary Enc", "高基数节省维度", "是", "否"],
        ["Ordinal Enc", "有序类别", "是", "是"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   colWidths=[0.32, 0.38, 0.15, 0.15])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white"); cell.set_edgecolor("#334")
    ax.set_title("类别编码方法总结", color="white", pad=10)

    # ── 子图9：QQ图（变换前后正态性比较） ───────────────────────
    ax = axes[2, 2]
    from scipy import stats
    (osm_orig, osr_orig), (slope, intercept, r) = stats.probplot(income_orig, dist="norm")
    (osm_pt, osr_pt), _ = stats.probplot(income_pt, dist="norm")
    ax.plot(osm_orig, osr_orig, ".", color="#e74c3c", alpha=0.4, markersize=3, label="原始")
    ax.plot(osm_pt, osr_pt, ".", color="#2ecc71", alpha=0.4, markersize=3, label="YeoJohnson")
    ax.plot(osm_pt, osm_pt, "--", color="#aaa", alpha=0.6, linewidth=1)
    ax.set_title("Q-Q 图：变换前后正态性对比", color="white", pad=8)
    ax.set_xlabel("Theoretical Quantiles", color="#aaa")
    ax.set_ylabel("Sample Quantiles", color="#aaa")
    ax.tick_params(colors="gray")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#444")

    plt.suptitle("特征编码方法全景", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("feature_encoding.png"))

    print("\n[DONE] 特征编码完成!")


if __name__ == "__main__":
    feature_encoding()
