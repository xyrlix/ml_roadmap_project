"""
网格搜索与随机搜索 (Grid & Random Search)
=========================================
基础超参数搜索方法对比

核心对比：
  - 网格搜索：穷举所有参数组合（穷尽但低效）
  - 随机搜索：随机采样参数（更高效，尤其在高维空间）
  - 实验表明：随机搜索在多数场景优于网格搜索（Bergstra & Bengio 2012）

实现内容：
  1. 网格搜索（GridSearchCV 风格）
  2. 随机搜索（RandomizedSearchCV 风格）
  3. 评估函数：合成分类任务（带超参数敏感性）
  4. 可视化：搜索轨迹、参数-性能热力图
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 搜索方法实现 ─────────────────────────

class GridSearch:
    """网格搜索：穷举所有参数组合"""
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def search(self, model, X_train, y_train, X_val, y_val):
        """搜索最优参数"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        # 生成所有组合
        import itertools
        combinations = list(itertools.product(*values))

        best_score = -np.inf
        best_params = {}
        results = []

        for combo in combinations:
            params = dict(zip(keys, combo))
            model.set_params(**params)
            model.fit(X_train, y_train)
            score = accuracy_score(y_val, model.predict(X_val))
            results.append({"params": params, "score": score})
            if score > best_score:
                best_score = score
                best_params = params.copy()

        return best_params, best_score, results


class RandomSearch:
    """随机搜索：随机采样参数"""
    def __init__(self, param_distributions, n_iter=20):
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def _sample(self, dist):
        """从分布采样参数"""
        if dist["type"] == "choice":
            return np.random.choice(dist["values"])
        elif dist["type"] == "uniform":
            return np.random.uniform(dist["low"], dist["high"])
        elif dist["type"] == "loguniform":
            return np.exp(np.random.uniform(np.log(dist["low"]), np.log(dist["high"])))
        else:
            raise ValueError(f"Unknown dist type: {dist}")

    def search(self, model, X_train, y_train, X_val, y_val, random_state=42):
        """随机搜索"""
        np.random.seed(random_state)
        best_score = -np.inf
        best_params = {}
        results = []

        for _ in range(self.n_iter):
            params = {k: self._sample(v) for k, v in self.param_distributions.items()}
            model.set_params(**params)
            model.fit(X_train, y_train)
            score = accuracy_score(y_val, model.predict(X_val))
            results.append({"params": params, "score": score})
            if score > best_score:
                best_score = score
                best_params = params.copy()

        return best_params, best_score, results


# ─────────────────────── 主程序 ──────────────────────────────────

def grid_random_search():
    print("网格搜索与随机搜索对比运行中...\n")

    # ── 生成合成数据 ────────────────────────────────────────────
    print("1. 生成合成分类数据...")
    X, y = make_classification(
        n_samples=800, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}")

    # ── 定义搜索空间 ─────────────────────────────────────────────
    # Random Forest 参数
    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10]
    }
    rf_param_dist = {
        "n_estimators": {"type": "choice", "values": [50, 100, 150, 200, 300, 400, 500]},
        "max_depth": {"type": "choice", "values": [3, 5, 7, 10, 15, 20, None]},
        "min_samples_split": {"type": "uniform", "low": 2, "high": 20}
    }

    # SVC 参数（惩罚系数 C，核函数 gamma）
    svc_param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1]
    }
    svc_param_dist = {
        "C": {"type": "loguniform", "low": 0.01, "high": 100},
        "gamma": {"type": "choice", "values": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10]}
    }

    # ── 执行搜索 ─────────────────────────────────────────────────────
    print("\n2. Random Forest 搜索...")
    rf_model = RandomForestClassifier(random_state=42)
    print("   网格搜索...")
    grid_rf = GridSearch(rf_param_grid)
    best_rf_grid, best_rf_score_grid, _ = grid_rf.search(rf_model, X_train, y_train, X_val, y_val)
    print(f"     最优参数: {best_rf_grid}, 评分: {best_rf_score_grid:.4f}")

    print("   随机搜索 (20次)...")
    random_rf = RandomSearch(rf_param_dist, n_iter=20)
    best_rf_rand, best_rf_score_rand, _ = random_rf.search(rf_model, X_train, y_train, X_val, y_val)
    print(f"     最优参数: {best_rf_rand}, 评分: {best_rf_score_rand:.4f}")

    # ── SVC 搜索 ─────────────────────────────────────────────────────
    print("\n3. SVC 搜索...")
    svc_model = SVC(kernel="rbf", random_state=42)
    print("   网格搜索...")
    grid_svc = GridSearch(svc_param_grid)
    best_svc_grid, best_svc_score_grid, _ = grid_svc.search(svc_model, X_train, y_train, X_val, y_val)
    print(f"     最优参数: {best_svc_grid}, 评分: {best_svc_score_grid:.4f}")

    print("   随机搜索 (20次)...")
    random_svc = RandomSearch(svc_param_dist, n_iter=20)
    best_svc_rand, best_svc_score_rand, _ = random_svc.search(svc_model, X_train, y_train, X_val, y_val)
    print(f"     最优参数: {best_svc_rand}, 评分: {best_svc_score_rand:.4f}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n4. 生成可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：Random Forest 搜索结果对比 ─────────────────────────────
    ax = axes[0, 0]
    methods = ["Grid Search", "Random Search"]
    scores_rf = [best_rf_score_grid, best_rf_score_rand]
    pal = ["#e74c3c", "#3498db"]
    bars = ax.bar(methods, scores_rf, color=pal, alpha=0.85)
    ax.set_title("Random Forest: 搜索方法准确率对比", color="white", pad=8)
    ax.set_ylabel("Validation Accuracy", color="#aaa")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, scores_rf):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=10)

    # ── 子图2：SVC 搜索结果对比 ───────────────────────────────────
    ax = axes[0, 1]
    scores_svc = [best_svc_score_grid, best_svc_score_rand]
    bars = ax.bar(methods, scores_svc, color=pal, alpha=0.85)
    ax.set_title("SVC: 搜索方法准确率对比", color="white", pad=8)
    ax.set_ylabel("Validation Accuracy", color="#aaa")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, scores_svc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=10)

    # ── 子图3：参数-性能热力图（RF） ────────────────────────────────
    ax = axes[1, 0]
    # 只展示 2 个参数的交互（n_estimators vs max_depth）
    n_est_vals = [50, 100, 200]
    depth_vals = [3, 5, 10, None]
    heatmap = np.zeros((len(n_est_vals), len(depth_vals)))
    for i, n_est in enumerate(n_est_vals):
        for j, depth in enumerate(depth_vals):
            rf_temp = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                         random_state=42)
            rf_temp.fit(X_train, y_train)
            heatmap[i, j] = accuracy_score(y_val, rf_temp.predict(X_val))

    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=heatmap.min(), vmax=heatmap.max())
    ax.set_xticks(range(len(depth_vals)))
    ax.set_xticklabels([str(d) for d in depth_vals], color="white")
    ax.set_yticks(range(len(n_est_vals)))
    ax.set_yticklabels([str(n) for n in n_est_vals], color="white")
    ax.set_title("RF: n_estimators vs max_depth", color="white", pad=8)
    ax.set_xlabel("max_depth", color="#aaa"); ax.set_ylabel("n_estimators", color="#aaa")
    for i in range(len(n_est_vals)):
        for j in range(len(depth_vals)):
            ax.text(j, i, f"{heatmap[i,j]:.2f}", ha="center", va="center",
                    color="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors="gray")

    # ── 子图4：方法对比总结表 ─────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["方法", "评估次数", "最优精度", "适用场景"],
        ["网格搜索", "累乘", f"{max(best_rf_score_grid, best_svc_score_grid):.3f}", "参数少"],
        ["随机搜索", "可控制", f"{max(best_rf_score_rand, best_svc_score_rand):.3f}", "参数多"],
        ["贝叶斯优化", "少", "- (需实现)", "昂贵黑盒"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   colWidths=[0.22, 0.18, 0.18, 0.42])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white"); cell.set_edgecolor("#334")
    ax.set_title("超参数优化方法对比", color="white", pad=10)

    plt.suptitle("网格搜索 vs 随机搜索对比", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("grid_random_search.png"))

    print("\n[DONE] 网格/随机搜索对比完成!")


if __name__ == "__main__":
    grid_random_search()
