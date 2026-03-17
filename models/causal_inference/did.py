# DID（差分中的差分）因果推断模型
# 通过双差分估计处理效应，消除时间趋势与组间差异

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import get_results_path, save_and_close


def did():
    """DID 差分中的差分因果推断实现（政策评估）"""
    print("DID 差分中的差分模型运行中...\n")

    # 1. 生成模拟数据（政策实验）
    print("1. 生成模拟政策实验数据...")
    np.random.seed(42)
    N_TREAT = 50        # 处理组人数
    N_CTRL = 50         # 对照组人数
    T_PRE  = 4          # 政策前观测期数
    T_POST = 4          # 政策后观测期数

    # 真实效应 = 5
    true_effect = 5.0
    beta_trend = 0.3      # 时间趋势

    # 处理组数据
    data_treat = []
    for i in range(N_TREAT):
        alpha_i = np.random.normal(10, 2)   # 个体固定效应
        for t in range(T_PRE + T_POST):
            policy = 1 if t >= T_PRE else 0
            y = alpha_i + beta_trend * t + policy * true_effect + \
                np.random.normal(0, 1.5)
            data_treat.append([i, 'Treat', t, policy, y])

    # 对照组数据（无处理效应）
    data_ctrl = []
    for i in range(N_CTRL):
        alpha_i = np.random.normal(10, 2)
        for t in range(T_PRE + T_POST):
            y = alpha_i + beta_trend * t + np.random.normal(0, 1.5)
            data_ctrl.append([i, 'Ctrl', t, 0, y])

    import pandas as pd
    df = pd.DataFrame(data_treat + data_ctrl,
                     columns=['id', 'group', 'time', 'policy', 'y'])
    print(f"   样本量: {len(df)}  真实处理效应: {true_effect}")

    # 2. 计算 DID
    print("2. 手动计算 DID...")
    mean_pre_treat = df[(df.group == 'Treat') & (df.time < T_PRE)].y.mean()
    mean_post_treat = df[(df.group == 'Treat') & (df.time >= T_PRE)].y.mean()
    mean_pre_ctrl  = df[(df.group == 'Ctrl')  & (df.time < T_PRE)].y.mean()
    mean_post_ctrl  = df[(df.group == 'Ctrl')  & (df.time >= T_PRE)].y.mean()

    delta_treat = mean_post_treat - mean_pre_treat
    delta_ctrl  = mean_post_ctrl - mean_pre_ctrl
    did_est = delta_treat - delta_ctrl

    print(f"   处理组均值变化: {delta_treat:.3f}")
    print(f"   对照组均值变化: {delta_ctrl:.3f}")
    print(f"   DID 估计: {did_est:.3f} (真实值={true_effect:.1f})")

    # 3. 回归 DID (OLS)
    print("3. OLS 回归 DID...")
    df['treat_post'] = (df.group == 'Treat') * (df.time >= T_PRE)
    import statsmodels.api as sm
    X = sm.add_constant(df[['group', 'time', 'treat_post']])
    y = df['y']
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['id']})
    print(model.summary().tables[1])

    # 4. 平行趋势检验（政策前对比）
    print("4. 平行趋势检验...")
    pre_treat_trend = []
    pre_ctrl_trend  = []
    for t in range(T_PRE):
        pre_treat_trend.append(
            df[(df.group == 'Treat') & (df.time == t)].y.mean())
        pre_ctrl_trend.append(
            df[(df.group == 'Ctrl') & (df.time == t)].y.mean())
    # 相关性检验
    corr, pval = stats.pearsonr(pre_treat_trend, pre_ctrl_trend)
    print(f"   政策前两组趋势相关系数: {corr:.3f} (p={pval:.3f})")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：DID 可视化（均值趋势）──
    ax = axes[0]
    treat_means = [df[(df.group == 'Treat') & (df.time == t)].y.mean()
                   for t in range(T_PRE + T_POST)]
    ctrl_means  = [df[(df.group == 'Ctrl') & (df.time == t)].y.mean()
                   for t in range(T_PRE + T_POST)]
    ax.plot(range(T_PRE + T_POST), treat_means, 'b-o',
            label='处理组', linewidth=1.5)
    ax.plot(range(T_PRE + T_POST), ctrl_means, 'r-s',
            label='对照组', linewidth=1.5)
    ax.axvline(x=T_PRE - 0.5, color='gray', linestyle='--',
               label='政策实施')
    ax.set_title('DID：处理组 vs 对照组均值趋势')
    ax.set_xlabel('时间')
    ax.set_ylabel('Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：平行趋势检验 ──
    ax = axes[1]
    ax.plot(range(T_PRE), pre_treat_trend, 'b-o', label='处理组')
    ax.plot(range(T_PRE), pre_ctrl_trend, 'r-s', label='对照组')
    ax.set_title('平行趋势检验（政策前）')
    ax.set_xlabel('时间')
    ax.set_ylabel('Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图3：DID 效应估计 ──
    ax = axes[2]
    methods = ['手动 DID', 'OLS 回归']
    estimates = [did_est, model.params['treat_post']]
    bars = ax.bar(methods, estimates,
                 color=['steelblue', 'darkorange'],
                 edgecolor='white', width=0.5)
    for bar, est in zip(bars, estimates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{est:.3f}', ha='center', fontsize=12)
    ax.axhline(y=true_effect, color='green', linestyle='--',
               linewidth=2, label=f'真实值={true_effect}')
    ax.set_title('DID 效应估计对比')
    ax.set_ylabel('Treatment Effect')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    save_path = get_results_path('did_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    did()
