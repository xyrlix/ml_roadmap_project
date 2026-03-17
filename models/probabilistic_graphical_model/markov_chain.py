# 马尔可夫链模型
# 状态序列满足马尔可夫性质：下一状态只取决于当前状态

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import get_results_path, save_and_close


def markov_chain():
    """马尔可夫链模型实现（天气预报 + 股市状态）"""
    print("马尔可夫链模型运行中...\n")

    # ── 例1：天气马尔可夫链 ───────────────────────────────────────────
    print("1. 天气马尔可夫链（晴/雨/阴）...")
    states      = ['晴天', '雨天', '阴天']
    n_states    = len(states)
    # 转移矩阵
    P = np.array([
        [0.60, 0.20, 0.20],   # 晴 -> 晴/雨/阴
        [0.25, 0.45, 0.30],   # 雨 -> ...
        [0.30, 0.30, 0.40],   # 阴 -> ...
    ])

    # 平稳分布（理论）
    eigvals, eigvecs = np.linalg.eig(P.T)
    stat_vec = np.real(eigvecs[:, np.isclose(eigvals, 1)][:, 0])
    stat_dist = np.abs(stat_vec) / np.abs(stat_vec).sum()
    print(f"   平稳分布（理论）: "
          + "  ".join(f"{s}={stat_dist[i]:.4f}" for i, s in enumerate(states)))

    # 蒙特卡洛模拟
    N_STEPS = 5000
    state_idx = 0   # 从晴天开始
    trajectory = [state_idx]
    for _ in range(N_STEPS):
        state_idx = np.random.choice(n_states, p=P[state_idx])
        trajectory.append(state_idx)
    empirical = np.array([(np.array(trajectory) == s).mean()
                          for s in range(n_states)])
    print(f"   平稳分布（模拟）: "
          + "  ".join(f"{s}={empirical[i]:.4f}" for i, s in enumerate(states)))

    # 多步转移矩阵 P^n
    P_power = {}
    for n in [1, 2, 5, 10, 20]:
        Pn = np.linalg.matrix_power(P, n)
        P_power[n] = Pn

    # ── 例2：股市状态马尔可夫链（牛/熊/盘整）──────────────────────────
    print("2. 股市马尔可夫链（牛市/熊市/盘整）...")
    market_states = ['牛市', '熊市', '盘整']
    Q = np.array([
        [0.70, 0.10, 0.20],
        [0.15, 0.65, 0.20],
        [0.30, 0.25, 0.45],
    ])
    # 模拟 500 天
    mktraj = [0]
    for _ in range(500):
        mktraj.append(np.random.choice(3, p=Q[mktraj[-1]]))
    mktraj = np.array(mktraj)

    # ── 可视化 ────────────────────────────────────────────────────────
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：天气转移矩阵热力图 ──
    ax = axes[0]
    im = ax.imshow(P, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f'{P[i,j]:.2f}', ha='center', va='center',
                    fontsize=12, color='black')
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_title('天气状态转移矩阵 P')
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')

    # ── 子图2：多步 P^n 从晴天出发的概率分布 ──
    ax = axes[1]
    steps_shown = sorted(P_power.keys())
    init = np.array([1, 0, 0])  # 从晴天出发
    for step in steps_shown:
        dist = init @ P_power[step]
        ax.plot(dist, marker='o', linewidth=1.5, label=f'n={step}')
    ax.plot(stat_dist, 'k--', linewidth=2, label='平稳分布')
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(states)
    ax.set_title('多步转移概率（从晴天出发）')
    ax.set_ylabel('Probability')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图3：股市状态轨迹 + 占比 ──
    ax = axes[2]
    ax_r = ax.twinx()
    n_show_market = 150
    t_axis = np.arange(n_show_market)
    # 轨迹
    ax.step(t_axis, mktraj[:n_show_market], color='steelblue',
            where='mid', linewidth=1.5, alpha=0.7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(market_states)
    ax.set_xlabel('Day')
    ax.set_ylabel('Market State', color='steelblue')
    # 占比条
    counts = [(mktraj == s).mean() for s in range(3)]
    bars = ax_r.bar(market_states, counts,
                    color=['steelblue', 'tomato', 'seagreen'],
                    alpha=0.4, width=0.3)
    for bar, cnt in zip(bars, counts):
        ax_r.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.01,
                  f'{cnt:.2f}', ha='center', fontsize=9)
    ax_r.set_ylabel('Frequency', color='gray')
    ax.set_title(f'股市状态轨迹（前 {n_show_market} 天）\n右轴=状态占比')

    save_path = get_results_path('markov_chain_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    markov_chain()
