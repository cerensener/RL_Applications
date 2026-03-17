# ============================================================
#  TD3 (Twin Delayed Deep Deterministic Policy Gradient)
#  Optimized for Academic Benchmark
#  Environment : Pendulum-v1  (Continuous Action Space)
#  Framework   : PyTorch
# ============================================================

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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("pip install optuna")

try:
    torch.use_deterministic_algorithms(True)
except:
    pass
    
# ============================================================
#  SEED CONTROL
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
#  REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, max_size: int = 100_000):
        self.memory = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# ============================================================
#  ACTOR  (Deterministic Policy)
# ============================================================

class ActorNet(nn.Module):

    def __init__(self, state_dim: int, action_dim: int,
                 action_min: float, action_max: float):
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max
        self.action_scale  = (action_max - action_min) / 2.0
        self.action_offset = (action_max + action_min) / 2.0

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.action_scale + self.action_offset

# ============================================================
#  CRITIC  (Twin Q-networks)
# ============================================================

class CriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        def build():
                return nn.Sequential(
                    nn.Linear(state_dim + action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                )
        
        self.q1 = build()
        self.q2 = build()

    def forward(self, state: torch.Tensor,
                action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


# ============================================================
#  TD3 AGENT
# ============================================================
class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int,
                 action_min: float, action_max: float,
                 lr_actor: float    = 1e-3,
                 lr_critic: float   = 1e-3,
                 gamma: float       = 0.99,
                 tau: float         = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float  = 0.5,
                 policy_freq: int   = 2,
                 batch_size: int    = 64,
                 memory_size: int   = 100_000,
                 exploration_noise: float = 0.1):

        self.action_min   = action_min
        self.action_max   = action_max
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq
        self.batch_size   = batch_size
        self.exploration_noise = exploration_noise
        self.update_count = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor        = ActorNet(state_dim, action_dim,
                                     action_min, action_max).to(self.device)
        self.actor_target = ActorNet(state_dim, action_dim,
                                     action_min, action_max).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic        = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(memory_size)

    # ── Action selection ───────────────────────────────────
    def get_action(self, state: np.ndarray,
                   exploration: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if exploration:
            noise = np.random.normal(0, self.exploration_noise, action.shape)
            action = np.clip(action + noise, self.action_min, self.action_max)

        return action

    # ── Learning step ──────────────────────────────────────
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.FloatTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Target policy smoothing
        with torch.no_grad():
            noise      = (torch.randn_like(a) * self.policy_noise).clamp(
                          -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(ns) + noise).clamp(
                           self.action_min, self.action_max)
            q1_next, q2_next = self.critic_target(ns, next_action)
            q_target = r + self.gamma * torch.min(q1_next, q2_next) * (1.0 - d)

        # Critic update
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # Delayed actor update
        actor_loss_val = None
        self.update_count += 1
        if self.update_count % self.policy_freq == 0:
            actor_loss = -self.critic.q1_only(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()

            # Soft update
            for tp, p in zip(self.actor_target.parameters(),
                             self.actor.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return actor_loss_val, critic_loss.item()

# ============================================================
#  TRAINING LOOP
# ============================================================
def train_td3(env, agent: TD3Agent,
              episodes: int,
              early_stop_window: int   = 20,
              verbose: bool            = True) -> dict:

    ep_rewards, actor_losses, critic_losses = [], [], []

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward   = 0.0
        ep_a_losses = []
        ep_c_losses = []
        done        = False

        while not done:
            action     = agent.get_action(obs, exploration=True)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated



            agent.replay_buffer.add(
                obs, action, reward, obs_next, float(terminated or truncated))

            a_loss, c_loss = agent.learn()
            if a_loss is not None:
                ep_a_losses.append(a_loss)
            if c_loss is not None:
                ep_c_losses.append(c_loss)

            ep_reward += float(reward)

            obs = obs_next

        ep_rewards.append(ep_reward)
        actor_losses.append(
            np.mean(ep_a_losses)  if ep_a_losses  else 0.0)
        critic_losses.append(
            np.mean(ep_c_losses) if ep_c_losses else 0.0)

        if verbose and episode % 10 == 0:
            avg = np.mean(ep_rewards[-early_stop_window:])
            print(f"  Ep {episode:4d} | Reward: {ep_reward:8.2f} | "
                  f"Avg{early_stop_window}: {avg:8.2f} | "
                  f"A_loss: {actor_losses[-1]:8.4f} | "
                  f"C_loss: {critic_losses[-1]:8.4f}")


    return {"rewards"      : ep_rewards,
            "actor_losses" : actor_losses,
            "critic_losses": critic_losses}

# ============================================================
#  ACADEMIC BENCHMARK PLOTTING
# ============================================================

def plot_academic(all_results: dict, smooth_window: int = 10,
                  goal: float = -100.0, save_path: str = None):
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(all_results)))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid') \
               if len(x) >= w else x

    for color, (label, res) in zip(colors, all_results.items()):
        r  = np.array(res["rewards"])
        al = np.array(res["actor_losses"])
        cl = np.array(res["critic_losses"])

        axes[0].plot(smooth(r,  smooth_window), label=label,
                     color=color, lw=1.6)
        axes[1].hist(r[-50:], bins=20, alpha=0.55,
                     label=label, color=color)
        axes[2].plot(smooth(al, smooth_window), color=color,
                     lw=1.4, label=label)
        axes[3].plot(smooth(cl, smooth_window), color=color,
                     lw=1.4, label=label)

    axes[0].axhline(goal, c='red', ls='--', lw=1.2,
                    label=f'Goal ({goal})')
    axes[1].axvline(goal, c='red', ls='--', lw=1.2,
                    label=f'Goal ({goal})')

    titles  = ["Smoothed Reward", "Reward Distribution (Last 50 ep)",
               "Actor Loss", "Critic Loss"]
    xlabels = ["Episode", "Reward", "Episode", "Episode"]
    ylabels = ["Reward", "Frequency", "Actor Loss", "Critic Loss"]

    for ax, t, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_title(t, fontsize=11, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("TD3 — Pendulum-v1 Benchmark",
                 fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Graphs saved to: {save_path}")
    plt.show()

# ============================================================
#  Statistical Evaluation  (Multi-seed)
# ============================================================

def multi_seed_eval(config: dict, seeds: list, episodes: int,
                    env_name: str = "Pendulum-v1") -> dict:
    all_rewards = []
    for seed in seeds:
        set_seed(seed)
        env = gym.make(env_name)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        agent = TD3Agent(
            state_dim    = env.observation_space.shape[0],
            action_dim   = env.action_space.shape[0],
            action_min   = float(env.action_space.low[0]),
            action_max   = float(env.action_space.high[0]),
            lr_actor     = config.get("lr_actor",     1e-3),
            lr_critic    = config.get("lr_critic",    1e-3),
            gamma        = config.get("gamma",        0.99),
            tau          = config.get("tau",          0.005),
            policy_noise = config.get("policy_noise", 0.2),
            noise_clip   = config.get("noise_clip",   0.5),
            policy_freq  = config.get("policy_freq",  2),
            batch_size   = config.get("batch_size",   64),
        )
        res = train_td3(env, agent, episodes=episodes, verbose=False)
        all_rewards.append(res["rewards"])
        env.close()
        print(f"    Seed {seed} → Mean reward: "
              f"{np.mean(res['rewards']):.2f}")

    max_len = max(len(r) for r in all_rewards)
    padded  = [r + [r[-1]] * (max_len - len(r)) for r in all_rewards]
    arr     = np.array(padded)
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

def optuna_objective(trial, env_name: str = "Pendulum-v1",
                     episodes: int = 300, seed: int = 42):
    set_seed(seed)

    lr_actor     = trial.suggest_float("lr_actor",     1e-4, 1e-2, log=True)
    lr_critic    = trial.suggest_float("lr_critic",    1e-4, 1e-2, log=True)
    gamma        = trial.suggest_float("gamma",        0.95, 0.999)
    tau          = trial.suggest_float("tau",          0.001, 0.02, log=True)
    policy_noise = trial.suggest_float("policy_noise", 0.1,  0.3)
    noise_clip   = trial.suggest_float("noise_clip",   0.3,  0.7)
    policy_freq  = trial.suggest_int  ("policy_freq",  1,    4)
    batch_size   = trial.suggest_categorical(
                       "batch_size", [64, 128, 256])

    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    agent = TD3Agent(
        state_dim    = env.observation_space.shape[0],
        action_dim   = env.action_space.shape[0],
        action_min   = float(env.action_space.low[0]),
        action_max   = float(env.action_space.high[0]),
        lr_actor     = lr_actor,
        lr_critic    = lr_critic,
        gamma        = gamma,
        tau          = tau,
        policy_noise = policy_noise,
        noise_clip   = noise_clip,
        policy_freq  = policy_freq,
        batch_size   = batch_size,
    )
    res = train_td3(env, agent, episodes=episodes, verbose=False)
    env.close()
    return np.mean(res["rewards"][-20:])

# ============================================================
#  MAIN  —  Benchmark, Optuna, Multi-seed Evaluation
# ============================================================

if __name__ == "__main__":

    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"  Outputs saved to: {OUTPUT_DIR}")

    ENV_NAME = "Pendulum-v1"
    EPISODES = 1000
    SEEDS    = [0, 1, 2, 3, 4]

    configs = [
        dict(policy_freq=2, batch_size=64,
             title="TD3 | freq=2, BS=64"),
        dict(policy_freq=2, batch_size=256,
             title="TD3 | freq=2, BS=256"),
    ]

    # ══════════════════════════════════════════════════════
    #  Phase 1 - Configuration Comparison (seed=42)
    # ══════════════════════════════════════════════════════

    print("\n" + "═"*60)
    print("  Phase 1: Configuration Comparison (seed=42)")
    print("═"*60)

    set_seed(42)
    all_results = {}

    for cfg in configs:
        print(f"\n  ▶ {cfg['title']}")
        env = gym.make(ENV_NAME)
        env.reset(seed=42)
        agent = TD3Agent(
            state_dim  = env.observation_space.shape[0],
            action_dim = env.action_space.shape[0],
            action_min = float(env.action_space.low[0]),
            action_max = float(env.action_space.high[0]),
            batch_size = cfg["batch_size"],
            policy_freq= cfg["policy_freq"],
        )
        res = train_td3(env, agent, episodes=EPISODES, verbose=True)
        all_results[cfg["title"]] = res
        env.close()

    plot_academic(all_results,
                  save_path=os.path.join(OUTPUT_DIR,
                                         "td3_benchmark.png"))

    print("\n" + "─"*60)
    print(f"{'Configuration':<30} {'Mean':>10} {'Max':>10} "
          f"{'Episodes':>9}")
    print("─"*60)
    for label, res in all_results.items():
        r = res["rewards"]
        print(f"  {label:<28} {np.mean(r):>10.2f} "
              f"{np.max(r):>10.2f} {len(r):>9d}")
    print("─"*60)

    # ══════════════════════════════════════════════════════
    #  Phase 2 - Optuna
    # ══════════════════════════════════════════════════════

    if OPTUNA_AVAILABLE:
        print("\n" + "═"*60)
        print("  Phase 2: Optuna Hyperparameter Optimization "
              "(50 trial)")
        print("═"*60)

        STUDY_DB   = os.path.join(OUTPUT_DIR, "td3_optuna.db")
        STUDY_NAME = "td3_pendulum"
        storage    = f"sqlite:///{STUDY_DB}"

        study = optuna.create_study(
            direction    = "maximize",
            study_name   = STUDY_NAME,
            storage      = storage,
            load_if_exists = True)

        remaining = 50 - len(study.trials)
        if remaining > 0:
            print(f" {remaining} trials executed...")
            study.optimize(
                lambda t: optuna_objective(t, episodes=300),
                n_trials          = remaining,
                show_progress_bar = True,
                catch             = (ValueError, RuntimeError))
        else:
            print(f"  [✓] Study already has 50+ trials. "
                  f"({len(study.trials)} trial)")

        print("\n  Best Parameters:")
        for k, v in study.best_params.items():
            print(f"    {k:<20}: {v}")
        print(f"  ✓ Best Mean Reward (last 20 eps): "
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
            plt.savefig(os.path.join(OUTPUT_DIR,
                                     "td3_optuna_results.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"  [INFO] Optuna graph error: {e}")

        best_cfg = study.best_params

    else:
        best_cfg = {
            "lr_actor": 1e-3, "lr_critic": 1e-3,
            "gamma": 0.99, "tau": 0.005,
            "policy_noise": 0.2, "noise_clip": 0.5,
            "policy_freq": 2, "batch_size": 64,
        }
        print("\n  [INFO] Optuna not available → using default best_cfg "
              "parameters.")

    # ══════════════════════════════════════════════════════
    #  Phase 3 - Multi-seed Evaluation
    # ══════════════════════════════════════════════════════

    print("\n" + "═"*60)
    print(f"  Phase 3: Multi-seed Evaluation ({SEEDS})")
    print("═"*60)

    multi_cfg = {
        "lr_actor"    : best_cfg.get("lr_actor",     1e-3),
        "lr_critic"   : best_cfg.get("lr_critic",    1e-3),
        "gamma"       : best_cfg.get("gamma",        0.99),
        "tau"         : best_cfg.get("tau",          0.005),
        "policy_noise": best_cfg.get("policy_noise", 0.2),
        "noise_clip"  : best_cfg.get("noise_clip",   0.5),
        "policy_freq" : best_cfg.get("policy_freq",  2),
        "batch_size"  : best_cfg.get("batch_size",   64),
    }

    print(f"\n  ▶ Best Config: {multi_cfg}")
    stats = multi_seed_eval(multi_cfg, seeds=SEEDS,
                            episodes=EPISODES, env_name=ENV_NAME)

    print(f"\n  Final Performance (last 20 eps, {len(SEEDS)} seeds):")
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

    ax.plot(x_s, m_s, color='darkorange', lw=2, label='Mean Reward')
    ax.fill_between(x_s, m_s - std_s, m_s + std_s,
                    alpha=0.25, color='darkorange', label='±1 Std')
    ax.axhline(-100, c='red', ls='--', lw=1.2, label='Goal (-100)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(
        f"TD3 Best Config — {len(SEEDS)} Seeds Mean ± Std",
        fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "td3_multiseed.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n Outputs saved: {OUTPUT_DIR}")
    print("  td3_benchmark.png       — Configuration comparison (seed=42)")
    print("  td3_optuna_results.png  — Hyperparameter Importances (Optuna)")
    print("  td3_multiseed.png       — Mean ± Std "
          "(Statistical Reliability)")