# DQN 深度Q网络强化学习模型
# 用神经网络逼近 Q 函数，结合经验回放和目标网络稳定训练

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import collections
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from utils import get_results_path, save_and_close


# ──────────────────────────────────────────────────────────
# 自定义 CartPole 环境（不依赖 gym，纯 numpy 实现）
# ──────────────────────────────────────────────────────────
class CartPole:
    """
    简化的 CartPole 物理仿真
    State: [cart_pos, cart_vel, pole_angle, pole_vel]
    Action: 0=向左推力, 1=向右推力
    """
    GRAVITY    = 9.8
    MASS_CART  = 1.0
    MASS_POLE  = 0.1
    POLE_LEN   = 0.5
    FORCE_MAG  = 10.0
    TAU        = 0.02

    POS_LIMIT   = 2.4
    ANGLE_LIMIT = 0.2094   # ≈12°

    def __init__(self):
        self.state = None

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, xdot, theta, thetadot = self.state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        total_mass = self.MASS_CART + self.MASS_POLE
        pole_mass_len = self.MASS_POLE * self.POLE_LEN

        temp = (force + pole_mass_len * thetadot**2 * sin_t) / total_mass
        theta_acc = ((self.GRAVITY * sin_t - cos_t * temp) /
                     (self.POLE_LEN * (4/3 - self.MASS_POLE * cos_t**2 / total_mass)))
        x_acc = temp - pole_mass_len * theta_acc * cos_t / total_mass

        x        += self.TAU * xdot
        xdot     += self.TAU * x_acc
        theta    += self.TAU * thetadot
        thetadot += self.TAU * theta_acc
        self.state = np.array([x, xdot, theta, thetadot])

        done = (abs(x) > self.POS_LIMIT or abs(theta) > self.ANGLE_LIMIT)
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done


# ──────────────────────────────────────────────────────────
# 经验回放缓冲区
# ──────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────────────────
# DQN 网络
# ──────────────────────────────────────────────────────────
def build_dqn(state_dim, n_actions):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(state_dim,)),
        Dense(64, activation='relu'),
        Dense(n_actions)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model


def dqn():
    """DQN 深度Q网络强化学习实现（CartPole 平衡）"""
    print("DQN 深度Q网络模型运行中...\n")

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    env = CartPole()
    STATE_DIM  = 4
    N_ACTIONS  = 2
    MAX_EP     = 500
    MAX_STEPS  = 200
    BATCH_SIZE = 64
    GAMMA      = 0.99
    EPS_START  = 1.0
    EPS_MIN    = 0.05
    EPS_DECAY  = 0.995
    TARGET_UPDATE_FREQ = 10    # 每 N 个 episode 更新目标网络

    # 主网络 & 目标网络
    online_net = build_dqn(STATE_DIM, N_ACTIONS)
    target_net = build_dqn(STATE_DIM, N_ACTIONS)
    target_net.set_weights(online_net.get_weights())

    buffer  = ReplayBuffer(capacity=20000)
    epsilon = EPS_START
    ep_rewards, ep_lengths = [], []

    print("1. 训练 DQN 智能体...")
    for ep in range(MAX_EP):
        state = env.reset()
        total_r = 0
        done    = False
        for step in range(MAX_STEPS):
            # ε-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                q_vals = online_net.predict(state[np.newaxis], verbose=0)[0]
                action = np.argmax(q_vals)

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))
            state   = next_state
            total_r += reward

            # 训练
            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = \
                    buffer.sample(BATCH_SIZE)
                states      = states.astype(np.float32)
                next_states = next_states.astype(np.float32)

                # Double DQN 目标
                next_actions = np.argmax(
                    online_net.predict(next_states, verbose=0), axis=1)
                next_q = target_net.predict(next_states, verbose=0)
                td_targets = rewards + GAMMA * \
                    next_q[np.arange(BATCH_SIZE), next_actions] * (1 - dones)

                current_q = online_net.predict(states, verbose=0)
                current_q[np.arange(BATCH_SIZE), actions.astype(int)] = \
                    td_targets.astype(np.float32)
                online_net.fit(states, current_q, verbose=0, batch_size=BATCH_SIZE)

            if done:
                break

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        ep_rewards.append(total_r)
        ep_lengths.append(step + 1)

        # 更新目标网络
        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.set_weights(online_net.get_weights())

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(ep_rewards[-50:])
            print(f"   Episode {ep+1:4d}  avg_steps(50)={avg_r:.1f}  "
                  f"epsilon={epsilon:.3f}")

    # 2. 评估
    print("2. 评估最终策略...")
    eval_rewards = []
    for _ in range(20):
        s = env.reset()
        total = 0
        for _ in range(MAX_STEPS):
            a = np.argmax(online_net.predict(s[np.newaxis], verbose=0)[0])
            s, r, done = env.step(a)
            total += r
            if done:
                break
        eval_rewards.append(total)
    print(f"   20 次评估平均步数: {np.mean(eval_rewards):.1f}")

    # 3. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    window = 20
    smooth = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')

    # ── 子图1：学习曲线 ──
    ax = axes[0]
    ax.plot(ep_rewards, alpha=0.3, color='steelblue', label='Episode Steps')
    ax.plot(np.arange(window-1, MAX_EP), smooth,
            color='steelblue', linewidth=2, label=f'{window}-ep 滑动均值')
    ax.axhline(y=MAX_STEPS, color='green', linestyle='--', label='Max steps')
    ax.set_title('DQN 学习曲线（CartPole）')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Steps (Balance Time)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：ε 衰减曲线 ──
    ax = axes[1]
    eps_curve = [max(EPS_MIN, EPS_START * EPS_DECAY**ep) for ep in range(MAX_EP)]
    ax.plot(eps_curve, color='darkorange', linewidth=2)
    ax.set_title('ε-greedy 衰减曲线')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.grid(True, alpha=0.3)

    # ── 子图3：评估分布 ──
    ax = axes[2]
    ax.hist(eval_rewards, bins=10, color='steelblue',
            edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(eval_rewards), color='red', linestyle='--',
               label=f'Mean={np.mean(eval_rewards):.1f}')
    ax.set_title('最终策略评估分布（20 次）')
    ax.set_xlabel('Episode Steps')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('dqn_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    dqn()
