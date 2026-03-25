# ============================================================
#  SAC (Soft Actor-Critic)
#  Optimized for Academic Benchmark
#  Environment : Pendulum-v1  (continuous action space)
#  Framework   : PyTorch
# ============================================================
#
#  Orijinal koddan yapılan düzeltmeler ve iyileştirmeler:
#   1. device global → agent içine taşındı
#   2. Reward scaling tutarsızlığı düzeltildi → (r+16)/16
#   3. goal=-150 → 3000 (scaled, PPO/TD3 ile tutarlı)
#   4. EPISODES=150 → 500
#   5. torch.use_deterministic_algorithms kaldırıldı
#   6. alpha_init parametresi artık log_alpha'ya doğru aktarılıyor
#   7. num_updates kullanılmıyor → kaldırıldı
#   8. done iki kez yazılmış → düzeltildi
#   9. Gradient clipping eklendi (actor + critic)
#  10. Optuna hiperparametre optimizasyonu eklendi
#  11. Çoklu seed istatistiksel değerlendirme eklendi
#  12. Early stopping eklendi
#  13. 3 aşamalı pipeline (PPO/TD3 ile tutarlı)

import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("pip install optuna")


# ============================================================
#  SEED KONTROLÜ
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


# ============================================================
#  REPLAY BUFFER
# ============================================================
class ReplayMemory:
    def __init__(self, state_dim: int, action_dim: int,
                 capacity: int):
        self.capacity = capacity
        self.ptr  = 0
        self.size = 0

        self.state      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.action     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.mask       = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, state, action, reward, next_state, mask):
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action
        self.reward[self.ptr]     = reward
        self.next_state[self.ptr] = next_state
        self.mask[self.ptr]       = mask
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (self.state[ind], self.action[ind],
                self.reward[ind], self.next_state[ind],
                self.mask[ind])

    def __len__(self):
        return self.size


# ============================================================
#  ACTOR  (stochastic policy with tanh squashing)
# ============================================================
class Actor(nn.Module):
    """
    Orijinal mimariden değişiklik yok.
    action_scale/bias register_buffer yerine device-aware
    tensor olarak tutulur.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int, action_high: float,
                 action_low: float, device: torch.device):
        super().__init__()
        self._device = device

        self.fc1      = nn.Linear(state_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head  = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(
            (action_high - action_low) / 2.0,
            dtype=torch.float32, device=device)
        self.action_bias  = torch.tensor(
            (action_high + action_low) / 2.0,
            dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu      = self.mu_head(x)
        log_std = torch.clamp(self.std_head(x), min=-20, max=2)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        mu, log_std = self.forward(state)
        std    = log_std.exp()
        normal = Normal(mu, std)

        # Reparameterization trick
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Tanh squashing log-prob correction
        log_prob  = normal.log_prob(x_t)
        log_prob -= torch.log(
            self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(1, keepdim=True)

        # Deterministic action (mean) — test fazı için
        det_action = (torch.tanh(mu) * self.action_scale
                      + self.action_bias)

        return action, log_prob, det_action


# ============================================================
#  CRITIC  (twin Q-networks)
# ============================================================
class Critic(nn.Module):
    """Orijinal mimariden değişiklik yok."""
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


# ============================================================
#  SAC AJANI
# ============================================================
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 action_high: float, action_low: float,
                 hidden_dim: int      = 256,
                 lr: float            = 3e-4,
                 gamma: float         = 0.99,
                 tau: float           = 0.005,
                 alpha_init: float    = 0.2,
                 batch_size: int      = 256,
                 memory_capacity: int = 100_000):

        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size

        # Networks
        self.actor = Actor(
            state_dim, action_dim, hidden_dim,
            action_high, action_low, self.device
        ).to(self.device)
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=lr)

        self.critic = Critic(
            state_dim, action_dim, hidden_dim
        ).to(self.device)
        self.critic_target = Critic(
            state_dim, action_dim, hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(
            self.critic.state_dict())
        self.critic_target.eval()
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        # FIX 6: alpha_init → log_alpha başlangıcına doğru aktarım
        self.target_entropy = -float(action_dim) * 0.5
        self.log_alpha = torch.tensor(
            [np.log(alpha_init)],
            requires_grad=True, dtype = torch.float32, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().detach()

        self.memory = ReplayMemory(
            state_dim, action_dim, memory_capacity)

    # ── Action selection ───────────────────────────────────
    def act(self, state: np.ndarray,
            deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, det = self.actor.sample(s)
        if deterministic:
            return det.cpu().numpy().flatten()
        return action.cpu().numpy().flatten()

    # ── Learning step ──────────────────────────────────────
    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None, None

        states, actions, rewards, next_states, masks = \
            self.memory.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.FloatTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(masks).to(self.device)

        # 1. Critic update
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(ns)
            q1_t, q2_t = self.critic_target(ns, next_a)
            min_q_t    = (torch.min(q1_t, q2_t)
                          - self.alpha * next_logp)
            q_target   = (r + self.gamma * (1.0 - d) * min_q_t).detach()

        q1, q2 = self.critic(s, a)
        critic_loss = (F.mse_loss(q1, q_target)
                       + F.mse_loss(q2, q_target))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # 2. Actor update
        a_pi, log_pi, _ = self.actor.sample(s)
        q1_pi, q2_pi    = self.critic(s, a_pi)
        min_q_pi        = torch.min(q1_pi, q2_pi)
        actor_loss      = (self.alpha * log_pi - min_q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # 3. Alpha update
        with torch.no_grad():
            _, log_pi_new, _ = self.actor.sample(s)
        alpha_loss = -(self.log_alpha * (log_pi_new + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().detach()

        # 4. Soft target update
        for tp, p in zip(self.critic_target.parameters(),
                         self.critic.parameters()):
            tp.data.copy_(
                self.tau * p.data
                + (1.0 - self.tau) * tp.data)

        return (actor_loss.item(),
                critic_loss.item(),
                alpha_loss.item())


# ============================================================
#  TRAINING LOOP
# ============================================================
def train_sac(env, agent: SACAgent,
              episodes: int,
              max_steps: int           = 3200,
              warmup_episodes: int     = 10,
              reward_scale: bool       = True,
              num_updates: int         = 1,
              early_stop_reward: float = 3100.0,
              early_stop_window: int   = 20,
              verbose: bool            = True) -> dict:

    ep_rewards, actor_losses, critic_losses = [], [], []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward   = 0.0
        ep_a_losses = []
        ep_c_losses = []
        done = False

        for t in range(max_steps):
            if done:
                break
            # Warmup: replay buffer yeterince dolana kadar random
            if episode <= warmup_episodes:
                action = env.action_space.sample()
            else:
                action = agent.act(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = \
                env.step(action)
            done = terminated or truncated

            # FIX 2: Reward scaling — PPO/TD3 ile tutarlı
            if reward_scale:
                stored_reward = float((reward + 16.0) / 16.0)
            else:
                stored_reward = float(reward)

            agent.memory.push(
                state, action, stored_reward,
                next_state, float(terminated))

            a_loss, c_loss, _ = agent.learn()
            if a_loss is not None:
                ep_a_losses.append(a_loss)
                ep_c_losses.append(c_loss)

            # ep_reward da scaled tutulur (grafik tutarlılığı)
            if reward_scale:
                ep_reward += float((reward + 16.0) / 16.0)
            else:
                ep_reward += float(reward)
            state = next_state

        ep_rewards.append(ep_reward)
        actor_losses.append(
            np.mean(ep_a_losses) if ep_a_losses else 0.0)
        critic_losses.append(
            np.mean(ep_c_losses) if ep_c_losses else 0.0)

        if verbose and episode % 10 == 0:
            avg = np.mean(ep_rewards[-early_stop_window:])
            print(f"  Ep {episode:4d} | Reward: {ep_reward:8.2f} | "
                  f"Avg{early_stop_window}: {avg:8.2f} | "
                  f"A_loss: {actor_losses[-1]:7.4f} | "
                  f"C_loss: {critic_losses[-1]:7.4f} | "
                  f"α: {agent.alpha.item():.4f}")

        if (len(ep_rewards) >= early_stop_window and
                np.mean(ep_rewards[-early_stop_window:])
                >= early_stop_reward):
            if verbose:
                print(f"\n  ✓ Early stop at episode {episode} "
                      f"(avg{early_stop_window} = "
                      f"{np.mean(ep_rewards[-early_stop_window:]):.2f})")
            break

    return {"rewards"      : ep_rewards,
            "actor_losses" : actor_losses,
            "critic_losses": critic_losses}


# ============================================================
#  AKADEMİK GRAFİK
# ============================================================
def plot_academic(all_results: dict, smooth_window: int = 10,
                  goal: float = 3100.0,
                  save_path: str = None):
    colors = plt.cm.tab10(
        np.linspace(0, 0.8, len(all_results)))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i // 2, i % 2])
            for i in range(4)]

    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid') \
               if len(x) >= w else x

    for color, (label, res) in zip(colors, all_results.items()):
        r  = np.array(res["rewards"])
        al = np.array(res["actor_losses"])
        cl = np.array(res["critic_losses"])

        axes[0].plot(smooth(r,  smooth_window),
                     label=label, color=color, lw=1.6)
        axes[1].hist(r[-50:], bins=20, alpha=0.55,
                     label=label, color=color)
        axes[2].plot(smooth(al, smooth_window),
                     color=color, lw=1.4, label=label)
        axes[3].plot(smooth(cl, smooth_window),
                     color=color, lw=1.4, label=label)

    axes[0].axhline(goal, c='red', ls='--', lw=1.2,
                    label=f'Goal ({goal})')
    axes[1].axvline(goal, c='red', ls='--', lw=1.2,
                    label=f'Goal ({goal})')

    titles  = ["Smoothed Reward",
               "Reward Distribution (Last 50 ep)",
               "Actor Loss", "Critic Loss"]
    xlabels = ["Episode", "Reward", "Episode", "Episode"]
    ylabels = ["Reward (scaled)", "Frequency",
               "Actor Loss", "Critic Loss"]

    for ax, t, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_title(t, fontsize=11, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("SAC — Pendulum-v1 Benchmark",
                 fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [✓] Grafik kaydedildi: {save_path}")
    plt.show()


# ============================================================
#  İSTATİSTİKSEL DEĞERLENDİRME  (Çoklu seed)
# ============================================================
def multi_seed_eval(config: dict, seeds: list,
                    episodes: int,
                    env_name: str = "Pendulum-v1") -> dict:
    all_rewards = []
    for seed in seeds:
        set_seed(seed)
        env = gym.make(env_name, max_episode_steps=3200)
        env.reset(seed=seed)
        agent = SACAgent(
            state_dim       = env.observation_space.shape[0],
            action_dim      = env.action_space.shape[0],
            action_high     = float(env.action_space.high[0]),
            action_low      = float(env.action_space.low[0]),
            hidden_dim      = config.get("hidden_dim",  256),
            lr              = config.get("lr",          3e-4),
            gamma           = config.get("gamma",       0.99),
            tau             = config.get("tau",         0.005),
            alpha_init      = config.get("alpha_init",  0.2),
            batch_size      = config.get("batch_size",  256),
        )
        res = train_sac(env, agent, episodes=episodes,
                        verbose=False)
        all_rewards.append(res["rewards"])
        env.close()
        print(f"    Seed {seed} → "
              f"Mean: {np.mean(res['rewards']):.2f}")

    max_len = max(len(r) for r in all_rewards)
    padded  = [r + [r[-1]] * (max_len - len(r))
               for r in all_rewards]
    arr = np.array(padded)
    return {
        "mean"      : arr.mean(axis=0),
        "std"       : arr.std(axis=0),
        "all"       : arr,
        "final_mean": arr[:, -20:].mean(),
        "final_std" : arr[:, -20:].std(),
    }


# ============================================================
#  OPTUNA
# ============================================================
def optuna_objective(trial,
                     env_name: str  = "Pendulum-v1",
                     episodes: int  = 300,
                     seed: int      = 42):
    set_seed(seed)

    lr         = trial.suggest_float("lr",        1e-4, 1e-2,
                                     log=True)
    hidden_dim = trial.suggest_categorical(
                     "hidden_dim", [128, 256, 512])
    gamma      = trial.suggest_float("gamma",     0.95, 0.999)
    tau        = trial.suggest_float("tau",       0.001, 0.02,
                                     log=True)
    alpha_init = trial.suggest_float("alpha_init", 0.05, 0.5)
    batch_size = trial.suggest_categorical(
                     "batch_size", [128, 256, 512])

    env = gym.make(env_name, max_episode_steps=3200)
    env.reset(seed=seed)
    agent = SACAgent(
        state_dim   = env.observation_space.shape[0],
        action_dim  = env.action_space.shape[0],
        action_high = float(env.action_space.high[0]),
        action_low  = float(env.action_space.low[0]),
        hidden_dim  = hidden_dim,
        lr          = lr,
        gamma       = gamma,
        tau         = tau,
        alpha_init  = alpha_init,
        batch_size  = batch_size,
    )
    res = train_sac(env, agent, episodes=episodes,
                    verbose=False)
    env.close()
    return np.mean(res["rewards"][-20:])


# ============================================================
#  MAIN  —  3 Aşamalı Pipeline
# ============================================================
if __name__ == "__main__":

    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"  [✓] Grafikler şuraya kaydedilecek: {OUTPUT_DIR}")

    ENV_NAME = "Pendulum-v1"
    EPISODES = 500
    SEEDS    = [0, 1, 2, 3, 4]

    configs = [
        dict(hidden_dim=256, batch_size=256,
             title="SAC | HD=256, BS=256"),
        dict(hidden_dim=256, batch_size=128,
             title="SAC | HD=256, BS=128"),
    ]

    # ══════════════════════════════════════════════════════
    #  AŞAMA 1 — Konfigürasyon Karşılaştırması (seed=42)
    # ══════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  AŞAMA 1: Konfigürasyon Karşılaştırması (seed=42)")
    print("═"*60)

    set_seed(42)
    all_results = {}

    for cfg in configs:
        print(f"\n  ▶ {cfg['title']}")
        env = gym.make(ENV_NAME, max_episode_steps=3200)
        env.reset(seed=42)
        agent = SACAgent(
            state_dim   = env.observation_space.shape[0],
            action_dim  = env.action_space.shape[0],
            action_high = float(env.action_space.high[0]),
            action_low  = float(env.action_space.low[0]),
            hidden_dim  = cfg["hidden_dim"],
            batch_size  = cfg["batch_size"],
        )
        res = train_sac(env, agent, episodes=EPISODES,
                        verbose=True)
        all_results[cfg["title"]] = res
        env.close()

    plot_academic(all_results,
                  save_path=os.path.join(
                      OUTPUT_DIR, "sac_benchmark.png"))

    print("\n" + "─"*60)
    print(f"{'Konfigürasyon':<30} {'Mean':>10} "
          f"{'Max':>10} {'Episodes':>9}")
    print("─"*60)
    for label, res in all_results.items():
        r = res["rewards"]
        print(f"  {label:<28} {np.mean(r):>10.2f} "
              f"{np.max(r):>10.2f} {len(r):>9d}")
    print("─"*60)

    # ══════════════════════════════════════════════════════
    #  AŞAMA 2 — Optuna
    # ══════════════════════════════════════════════════════
    if OPTUNA_AVAILABLE:
        print("\n" + "═"*60)
        print("  AŞAMA 2: Optuna Hiperparametre Optimizasyonu "
              "(50 trial)")
        print("═"*60)

        STUDY_DB   = os.path.join(OUTPUT_DIR, "sac_optuna.db")
        STUDY_NAME = "sac_pendulum"
        storage    = f"sqlite:///{STUDY_DB}"

        study = optuna.create_study(
            direction      = "maximize",
            study_name     = STUDY_NAME,
            storage        = storage,
            load_if_exists = True)

        remaining = 50 - len(study.trials)
        if remaining > 0:
            print(f"  [→] {remaining} trial çalıştırılıyor...")
            study.optimize(
                lambda t: optuna_objective(t, episodes=300),
                n_trials          = remaining,
                show_progress_bar = True,
                catch             = (ValueError, RuntimeError))
        else:
            print(f"  [✓] Study zaten tamamlanmış "
                  f"({len(study.trials)} trial)")

        print("\n  ✓ En İyi Hiperparametreler:")
        for k, v in study.best_params.items():
            print(f"    {k:<20}: {v}")
        print(f"  ✓ En İyi Mean Reward (son 20 ep): "
              f"{study.best_value:.2f}")

        try:
            from optuna.visualization.matplotlib import (
                plot_param_importances,
                plot_optimization_history)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            plt.sca(axes[0])
            plot_optimization_history(study)
            axes[0].set_title("Optimization History")
            plt.sca(axes[1])
            plot_param_importances(study)
            axes[1].set_title("Parameter Importances")
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR,
                             "sac_optuna_results.png"),
                dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"  [INFO] Optuna grafik hatası: {e}")

        best_cfg = study.best_params

    else:
        best_cfg = {
            "lr": 3e-4, "hidden_dim": 256, "gamma": 0.99,
            "tau": 0.005, "alpha_init": 0.2, "batch_size": 256,
        }
        print("\n  [INFO] Optuna yok → varsayılan best_cfg "
              "kullanılıyor.")

    # ══════════════════════════════════════════════════════
    #  AŞAMA 3 — Çoklu Seed
    # ══════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print(f"  AŞAMA 3: Çoklu Seed Değerlendirme ({SEEDS})")
    print("═"*60)

    multi_cfg = {
        "lr"        : best_cfg.get("lr",         3e-4),
        "hidden_dim": best_cfg.get("hidden_dim",  256),
        "gamma"     : best_cfg.get("gamma",       0.99),
        "tau"       : best_cfg.get("tau",         0.005),
        "alpha_init": best_cfg.get("alpha_init",  0.2),
        "batch_size": best_cfg.get("batch_size",  256),
    }

    print(f"\n  ▶ Best Config: {multi_cfg}")
    stats = multi_seed_eval(multi_cfg, seeds=SEEDS,
                            episodes=EPISODES,
                            env_name=ENV_NAME)

    print(f"\n  ✓ Final Performance (son 20 ep, "
          f"{len(SEEDS)} seed):")
    print(f"    Mean ± Std : "
          f"{stats['final_mean']:.2f} ± {stats['final_std']:.2f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    smooth_w = 10
    def sm(a):
        return np.convolve(a, np.ones(smooth_w) / smooth_w,
                           mode='valid')
    m_s   = sm(stats["mean"])
    std_s = sm(stats["std"])
    x_s   = np.arange(len(m_s))

    ax.plot(x_s, m_s, color='seagreen', lw=2,
            label='Mean Reward')
    ax.fill_between(x_s, m_s - std_s, m_s + std_s,
                    alpha=0.25, color='seagreen',
                    label='±1 Std')
    ax.axhline(3100, c='red', ls='--', lw=1.2,
               label='Goal (3100)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (scaled)")
    ax.set_title(
        f"SAC Best Config — {len(SEEDS)} Seeds Mean ± Std",
        fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sac_multiseed.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n  Tüm çıktılar kaydedildi: {OUTPUT_DIR}")
    print("  sac_benchmark.png       — Konfigürasyon karşılaştırması")
    print("  sac_optuna_results.png  — Hiperparametre önemi")
    print("  sac_multiseed.png       — Mean ± Std "
          "(istatistiksel güvenilirlik)")