# PPO 近端策略优化强化学习模型
# Actor-Critic 框架 + Clipped Surrogate Objective，稳定策略梯度训练

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from utils import get_results_path, save_and_close


# ──────────────────────────────────────────────────────────
# 复用 CartPole 环境
# ──────────────────────────────────────────────────────────
class CartPole:
    GRAVITY = 9.8; MASS_CART = 1.0; MASS_POLE = 0.1
    POLE_LEN = 0.5; FORCE_MAG = 10.0; TAU = 0.02
    POS_LIMIT = 2.4; ANGLE_LIMIT = 0.2094

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
        pml = self.MASS_POLE * self.POLE_LEN
        temp = (force + pml * thetadot**2 * sin_t) / total_mass
        theta_acc = (self.GRAVITY * sin_t - cos_t * temp) / \
                    (self.POLE_LEN * (4/3 - self.MASS_POLE * cos_t**2 / total_mass))
        x_acc = temp - pml * theta_acc * cos_t / total_mass
        x += self.TAU * xdot; xdot += self.TAU * x_acc
        theta += self.TAU * thetadot; thetadot += self.TAU * theta_acc
        self.state = np.array([x, xdot, theta, thetadot])
        done = abs(x) > self.POS_LIMIT or abs(theta) > self.ANGLE_LIMIT
        return self.state.copy(), 1.0 if not done else 0.0, done


# ──────────────────────────────────────────────────────────
# Actor-Critic 网络
# ──────────────────────────────────────────────────────────
def build_actor_critic(state_dim, n_actions):
    inp = Input(shape=(state_dim,))
    shared = Dense(64, activation='relu')(inp)
    shared = Dense(64, activation='relu')(shared)
    # Actor: 输出动作概率
    actor_out = Dense(n_actions, activation='softmax')(shared)
    # Critic: 输出状态价值
    critic_out = Dense(1)(shared)
    actor  = Model(inp, actor_out, name='Actor')
    critic = Model(inp, critic_out, name='Critic')
    return actor, critic


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """广义优势估计（GAE）"""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv   = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:-1]
    return advantages, returns


def ppo():
    """PPO 近端策略优化实现（CartPole）"""
    print("PPO 近端策略优化模型运行中...\n")

    tf.random.set_seed(42)
    np.random.seed(42)

    env = CartPole()
    STATE_DIM  = 4
    N_ACTIONS  = 2
    LR_ACTOR   = 3e-4
    LR_CRITIC  = 1e-3
    GAMMA      = 0.99
    LAM        = 0.95
    CLIP_EPS   = 0.2
    EPOCHS_PPO = 4         # 每次更新的 PPO epochs
    BATCH_SIZE = 64
    ROLLOUT_LEN = 512      # 每次收集的步数
    TOTAL_STEPS = 80000
    MAX_EP_LEN  = 200

    actor, critic = build_actor_critic(STATE_DIM, N_ACTIONS)
    actor_opt  = Adam(learning_rate=LR_ACTOR)
    critic_opt = Adam(learning_rate=LR_CRITIC)

    print("1. 训练 PPO 智能体...")
    total_steps = 0
    update_count = 0
    ep_rewards_log = []
    update_rewards = []

    state = env.reset()
    ep_r  = 0
    ep_steps = 0
    current_ep_rewards = []

    while total_steps < TOTAL_STEPS:
        # ── 收集 Rollout ───────────────────────────────
        states, actions, rewards, dones, log_probs_old, values = \
            [], [], [], [], [], []

        for _ in range(ROLLOUT_LEN):
            s_tensor = tf.constant(state[np.newaxis], dtype=tf.float32)
            probs = actor(s_tensor, training=False).numpy()[0]
            action = np.random.choice(N_ACTIONS, p=probs)
            lp = np.log(probs[action] + 1e-8)

            next_state, reward, done = env.step(action)
            v = critic(s_tensor, training=False).numpy()[0, 0]

            states.append(state); actions.append(action)
            rewards.append(reward); dones.append(float(done))
            log_probs_old.append(lp); values.append(v)

            state  = next_state if not done else env.reset()
            ep_r  += reward
            ep_steps += 1
            total_steps += 1

            if done:
                current_ep_rewards.append(ep_r)
                ep_r, ep_steps = 0, 0

        # 计算最后一步的价值
        v_next = critic(tf.constant(state[np.newaxis], dtype=tf.float32),
                        training=False).numpy()[0, 0]
        values_ext = np.array(values + [v_next], dtype=np.float32)
        advantages, returns = compute_gae(
            np.array(rewards, np.float32),
            values_ext,
            np.array(dones, np.float32), GAMMA, LAM)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO 更新 ────────────────────────────────────
        states_arr   = np.array(states, np.float32)
        actions_arr  = np.array(actions, np.int32)
        lp_old_arr   = np.array(log_probs_old, np.float32)

        indices = np.arange(ROLLOUT_LEN)
        for _ in range(EPOCHS_PPO):
            np.random.shuffle(indices)
            for start in range(0, ROLLOUT_LEN, BATCH_SIZE):
                idx = indices[start:start+BATCH_SIZE]
                sb = tf.constant(states_arr[idx])
                ab = actions_arr[idx]
                lp_old_b = lp_old_arr[idx]
                adv_b    = tf.constant(advantages[idx])
                ret_b    = tf.constant(returns[idx])

                # Actor 更新
                with tf.GradientTape() as tape:
                    probs_new = actor(sb, training=True)
                    lp_new = tf.math.log(
                        tf.gather(probs_new, ab, batch_dims=1) + 1e-8)
                    ratio  = tf.exp(lp_new - tf.cast(lp_old_b, tf.float32))
                    clip_r = tf.clip_by_value(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(ratio * adv_b, clip_r * adv_b))
                    # Entropy bonus
                    entropy = -tf.reduce_mean(
                        tf.reduce_sum(probs_new * tf.math.log(probs_new + 1e-8),
                                      axis=1))
                    actor_loss -= 0.01 * entropy

                grads = tape.gradient(actor_loss, actor.trainable_variables)
                actor_opt.apply_gradients(
                    zip(grads, actor.trainable_variables))

                # Critic 更新
                with tf.GradientTape() as tape:
                    v_pred = critic(sb, training=True)[:, 0]
                    critic_loss = tf.reduce_mean(
                        tf.square(ret_b - v_pred))
                grads = tape.gradient(critic_loss, critic.trainable_variables)
                critic_opt.apply_gradients(
                    zip(grads, critic.trainable_variables))

        update_count += 1
        if current_ep_rewards:
            avg = np.mean(current_ep_rewards[-20:])
            update_rewards.append(avg)
            ep_rewards_log.extend(current_ep_rewards)
            current_ep_rewards = []

        if update_count % 10 == 0:
            print(f"   Update {update_count:4d}  total_steps={total_steps:6d}  "
                  f"avg_steps(20)={update_rewards[-1]:.1f}")

    # 2. 评估
    print("2. 评估最终策略...")
    eval_rs = []
    for _ in range(20):
        s = env.reset(); total = 0
        for _ in range(MAX_EP_LEN):
            p = actor(tf.constant(s[np.newaxis], dtype=tf.float32),
                      training=False).numpy()[0]
            a = np.argmax(p)
            s, r, done = env.step(a)
            total += r
            if done: break
        eval_rs.append(total)
    print(f"   20 次评估平均步数: {np.mean(eval_rs):.1f}")

    # 3. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：完整学习曲线 ──
    ax = axes[0]
    if ep_rewards_log:
        win = 20
        sm = np.convolve(ep_rewards_log, np.ones(win)/win, mode='valid')
        ax.plot(ep_rewards_log, alpha=0.2, color='steelblue')
        ax.plot(np.arange(win-1, len(ep_rewards_log)), sm,
                color='steelblue', linewidth=2)
    ax.axhline(y=MAX_EP_LEN, color='green', linestyle='--', label='Max steps')
    ax.set_title('PPO 学习曲线（CartPole）')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：每次 Update 平均奖励 ──
    ax = axes[1]
    ax.plot(update_rewards, color='darkorange', linewidth=2)
    ax.set_title('PPO Update 奖励趋势')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Avg Steps (last 20 eps)')
    ax.grid(True, alpha=0.3)

    # ── 子图3：评估分布 ──
    ax = axes[2]
    ax.hist(eval_rs, bins=10, color='seagreen', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(eval_rs), color='red', linestyle='--',
               label=f'Mean={np.mean(eval_rs):.1f}')
    ax.set_title('最终策略评估分布（20 次）')
    ax.set_xlabel('Steps Balanced')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('ppo_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    ppo()
