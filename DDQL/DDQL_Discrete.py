# ============================================================
#  Double DQN — Optimized for Academic Benchmark
#  Environment : CartPole-v1
#  Framework   : PyTorch
# ============================================================

from collections import deque
import random
import time
import copy
import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# OPTUNA
# ============================================================
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("pip install optuna")

# ============================================================
#  SEED CONTROL (Critical for academic reproducibility)
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
#  NEURAL NETWORK
# ============================================================
class QNetwork(nn.Module):
    """
    Dueling:
      - Dueling=False → standard Q-network
      - Dueling=True  → V(s) + A(s,a) 
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 64, dueling: bool = False):
        super().__init__()
        self.dueling = dueling

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.01),
        )

        if dueling:
            self.value_stream = nn.Linear(hidden_dim * 2, 1)
            self.adv_stream   = nn.Linear(hidden_dim * 2, action_dim)
        else:
            self.out = nn.Linear(hidden_dim * 2, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        if self.dueling:
            v   = self.value_stream(feat)                     # (B,1)
            adv = self.adv_stream(feat)                       # (B,A)
            return v + adv - adv.mean(dim=1, keepdim=True)   # Q(s,a)
        return self.out(feat)


# ============================================================
#  DDQN AGENT  (Vectorized Replay + Gradient Clipping)
# ============================================================
class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 64, lr: float = 1e-3,
                 dueling: bool = False,
                 device: str = "cpu"):

        self.action_dim = action_dim
        self.device     = torch.device(device)

        self.model  = QNetwork(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.hard_update()
        self.target.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # ── HuberLoss (SmoothL1) —
        self.criterion = nn.SmoothL1Loss()

    #───────────────────Inference─────────────────────────────────────────
    def predict(self, state: np.ndarray) -> torch.Tensor:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(s).squeeze(0)

    def target_predict(self, state: np.ndarray) -> torch.Tensor:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.target(s).squeeze(0)

    #───────────────────Batch Update───────────────────────
    def update(self, states, actions, td_targets):

        s  = torch.FloatTensor(np.array(states)).to(self.device)   # (B, state_dim)
        a  = torch.LongTensor(actions).to(self.device)             # (B,)
        tg = torch.FloatTensor(td_targets).to(self.device)        # (B,)

        q_pred = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)
        loss   = self.criterion(q_pred, tg)

        self.optimizer.zero_grad()
        loss.backward()
        # ── Gradient clipping ──
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    #───────────────────Vectorized Replay───────────────────
    def replay(self, memory, batch_size: int, gamma: float = 0.99):

        if len(memory) < batch_size:
            return None

        batch      = random.sample(memory, batch_size)
        states     = np.array([e[0] for e in batch], dtype=np.float32)
        actions    = np.array([e[1] for e in batch], dtype=np.int64)
        next_states= np.array([e[2] for e in batch], dtype=np.float32)
        rewards    = np.array([e[3] for e in batch], dtype=np.float32)
        dones      = np.array([e[4] for e in batch], dtype=np.float32)

        # Double DQN: online net → best action, target net → Q value
        ns_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            best_actions  = self.model(ns_tensor).argmax(dim=1)        # online
            next_q        = self.target(ns_tensor)                     # target
            next_q_values = next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1).cpu().numpy()

        td_targets = rewards + gamma * next_q_values * (1.0 - dones)
        return self.update(states, actions, td_targets)

    #───────────────────Network Sync───────────────────
    def soft_update(self, tau: float = 0.01):
        for tp, mp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(tau * mp.data + (1.0 - tau) * tp.data)

    def hard_update(self):
        self.target.load_state_dict(self.model.state_dict())

    # ── Mean Q (monitoring) ────────────────────────────────
    def mean_q(self, memory, n: int = 256) -> float:
        if len(memory) < n:
            return 0.0
        sample = random.sample(memory, min(n, len(memory)))
        states = torch.FloatTensor(np.array([e[0] for e in sample])).to(self.device)
        with torch.no_grad():
            return self.model(states).max(dim=1).values.mean().item()

# ============================================================
#  TRAINING LOOP
# ============================================================

def train_ddqn(env, agent, episodes: int,
               gamma: float         = 0.99,
               epsilon: float       = 1.0,
               epsilon_min: float   = 0.01,
               epsilon_decay: float = 0.9995,   # ← step-based decay
               replay_size: int     = 128,
               memory_size: int     = 10_000,
               min_memory: int      = 1_000,    # ← warmup:
               use_replay: bool     = True,
               use_soft_update: bool = True,
               tau: float           = 0.005,    # ← more stabilized
               hard_update_freq: int = 500,
               early_stop_reward: float = 475.0,
               early_stop_window: int   = 50,
               verbose: bool = True,
               title: str = "DDQN") -> dict:

    memory  = deque(maxlen=memory_size)
    rewards, losses, mean_qs, epsilons = [], [], [], []
    global_step = 0

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done, total_reward, ep_losses = False, 0.0, []

        while not done:
            # ε-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.predict(state).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            memory.append((state, action, next_state, reward, float(terminated)))

            # Warmup
            if len(memory) >= min_memory:
                if use_replay:
                    loss = agent.replay(memory, replay_size, gamma)
                else:
                    # Single-step update (no replay)
                    s, a, ns, r, term = memory[-1]
                    if term:
                        td = np.array([r], dtype=np.float32)
                    else:
                        ba = agent.predict(ns).argmax().item()
                        nq = agent.target_predict(ns)[ba].item()
                        td = np.array([r + gamma * nq], dtype=np.float32)
                    loss = agent.update([s], [a], td)
                if loss is not None:
                    ep_losses.append(loss)

            # Step-based hard update
            if not use_soft_update and global_step % hard_update_freq == 0:
                agent.hard_update()
            if use_soft_update:
                agent.soft_update(tau)

            # ← Step-based epsilon decay (not an episode!)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            state = next_state
            global_step += 1

        rewards.append(total_reward)
        losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        mean_qs.append(agent.mean_q(memory))
        epsilons.append(epsilon)

        if verbose and episode % 50 == 0:
            avg = np.mean(rewards[-early_stop_window:])
            print(f"  Ep {episode:4d} | Reward: {total_reward:6.1f} | "
                  f"Avg{early_stop_window}: {avg:6.1f} | "
                  f"Loss: {losses[-1]:.4f} | ε: {epsilon:.4f} | "
                  f"Mem: {len(memory)}")

        # Early stopping
        if (len(rewards) >= early_stop_window and
                np.mean(rewards[-early_stop_window:]) >= early_stop_reward):
            if verbose:
                print(f"\n  ✓ Early stop at episode {episode} "
                      f"(avg{early_stop_window} = {np.mean(rewards[-early_stop_window:]):.1f})")
            break

    return {"rewards": rewards, "losses": losses,
            "mean_qs": mean_qs, "epsilons": epsilons,
            "title": title}


# ============================================================
#  ACADEMICAL GRAPH
# ============================================================

def plot_academic(all_results: dict, smooth_window: int = 20,
                  goal: float = 475.0, save_path: str = None):
    configs = list(all_results.keys())
    colors  = plt.cm.tab10(np.linspace(0, 0.8, len(configs)))

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid') if len(x) >= w else x

    for color, (label, res) in zip(colors, all_results.items()):
        r  = np.array(res["rewards"])
        l  = np.array(res["losses"])
        mq = np.array(res["mean_qs"])

        sr = smooth(r, smooth_window)
        axes[0].plot(sr, label=label, color=color, lw=1.6)
        axes[1].hist(r[-100:], bins=20, alpha=0.55, label=label, color=color)
        axes[2].plot(smooth(l, smooth_window), color=color, lw=1.4, label=label)
        axes[3].plot(smooth(mq, smooth_window), color=color, lw=1.4, label=label)

    axes[0].axhline(goal, c='red', ls='--', lw=1.2, label=f'Goal ({goal})')
    axes[1].axvline(goal, c='red', ls='--', lw=1.2, label=f'Goal ({goal})')

    titles  = ["Smoothed Reward", "Reward Distribution (Last 100 ep)",
               "Training Loss", "Mean Q-Value"]
    xlabels = ["Episode", "Reward", "Episode", "Episode"]
    ylabels = ["Reward", "Frequency", "Loss (Huber)", "Mean Q"]

    for ax, t, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_title(t, fontsize=11, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Double DQN — CartPole-v1 Benchmark",
                 fontsize=14, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Graphic saved: {save_path}")
    plt.show()


# ============================================================
# STATISTICAL EVALUATION WITH MULTIPLE SEEDS
# ============================================================

def multi_seed_eval(config: dict, seeds: list, episodes: int,
                    env_name: str = "CartPole-v1") -> dict:

    #It executes across multiple seeds and computes the mean $\pm$ standard deviation;
    # a procedure essential for ensuring statistical reliability in academic publications.
    all_rewards = []
    for seed in seeds:
        set_seed(seed)
        env   = gym.make(env_name)
        env.reset(seed=seed)
        agent = DDQNAgent(
            state_dim  = env.observation_space.shape[0],
            action_dim = int(env.action_space.n),
            hidden_dim = config.get("hidden_dim", 64),
            lr         = config.get("lr", 1e-3),
            dueling    = config.get("dueling", False),
        )
        res = train_ddqn(env, agent, episodes=episodes, verbose=False,
                         **{k: v for k, v in config.items()
                            if k not in ("hidden_dim", "lr", "dueling", "title")})
        all_rewards.append(res["rewards"])
        env.close()
        print(f"    Seed {seed} → Mean reward: {np.mean(res['rewards']):.1f}")

    #Padding sequences of varying lengths
    max_len = max(len(r) for r in all_rewards)
    padded  = [r + [r[-1]] * (max_len - len(r)) for r in all_rewards]
    arr     = np.array(padded)
    return {
        "mean"   : arr.mean(axis=0),
        "std"    : arr.std(axis=0),
        "all"    : arr,
        "final_mean": arr[:, -50:].mean(),
        "final_std" : arr[:, -50:].std(),
    }

# ============================================================
# Optuna-based Hyperparameter Optimization
# ============================================================

def optuna_objective(trial, env_name: str = "CartPole-v1",
                     episodes: int = 300, seed: int = 42):
    set_seed(seed)

    lr            = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
    hidden_dim    = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    gamma         = trial.suggest_float("gamma",        0.90, 0.999)
    epsilon_decay = trial.suggest_float("epsilon_decay",0.990, 0.9995)
    tau           = trial.suggest_float("tau",          0.001, 0.1, log=True)
    replay_size   = trial.suggest_categorical("replay_size", [32, 64, 128, 256])
    dueling       = trial.suggest_categorical("dueling", [True, False])

    env   = gym.make(env_name)
    env.reset(seed=seed)
    agent = DDQNAgent(
        state_dim  = env.observation_space.shape[0],
        action_dim = int(env.action_space.n),
        hidden_dim = hidden_dim,
        lr         = lr,
        dueling    = dueling,
    )
    res = train_ddqn(env, agent, episodes=episodes,
                     gamma=gamma, epsilon_decay=epsilon_decay,
                     tau=tau, replay_size=replay_size,
                     use_soft_update=True, verbose=False)
    env.close()
    return np.mean(res["rewards"][-50:])


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    # ── Output Dir ───────────────
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"  Graphs will be saved: {OUTPUT_DIR}")

    ENV_NAME = "CartPole-v1"
    EPISODES = 500
    SEEDS    = [0, 1, 2, 3, 4]   # For academic reproducibility: 5 seeds

    #───────────────────Configurations───────────────────

    configs = [
        dict(use_replay=True,  use_soft_update=True,
             tau=0.005, title="DDQN | Replay + Soft Update"),
        dict(use_replay=True,  use_soft_update=False,
             hard_update_freq=500, title="DDQN | Replay + Hard Update"),
        dict(use_replay=False, use_soft_update=True,
             tau=0.005, title="DDQN | No Replay + Soft Update"),
        dict(use_replay=False, use_soft_update=False,
             hard_update_freq=500, title="DDQN | No Replay + Hard Update"),
    ]

    # ══════════════════════════════════════════════════════
    # PHASE 1 — Single-seed Fast Benchmarking
    # ══════════════════════════════════════════════════════

    print("\n" + "═"*60)
    print("  PHASE 1 — Single-seed Fast Benchmarking (seed=42)")
    print("═"*60)

    set_seed(42)
    all_results = {}

    for cfg in configs:
        print(f"\n  ▶ {cfg['title']}")
        env   = gym.make(ENV_NAME)
        env.reset(seed=42)
        agent = DDQNAgent(
            state_dim  = env.observation_space.shape[0],
            action_dim = int(env.action_space.n),
            hidden_dim = 128,
            lr         = 1e-4,
        )
        res = train_ddqn(env, agent, episodes=EPISODES, verbose=True,
                         **{k: v for k, v in cfg.items() if k != "title"},
                         title=cfg["title"])
        all_results[cfg["title"]] = res
        env.close()

    plot_academic(all_results,
                  save_path=os.path.join(OUTPUT_DIR, "ddqn_benchmark.png"))

    #───────────────────Summary Table───────────────────
    print("\n" + "─"*60)
    print(f"{'Konfigürasyon':<40} {'Mean':>8} {'Max':>8} {'Episodes':>9}")
    print("─"*60)
    for label, res in all_results.items():
        r = res["rewards"]
        print(f"  {label:<38} {np.mean(r):>8.1f} {np.max(r):>8.1f} {len(r):>9d}")
    print("─"*60)
    
    # ══════════════════════════════════════════════════════
    #  PHASE 2 — Optuna Hyperparameter Optimization (50 trials)
    # ══════════════════════════════════════════════════════
    
    if OPTUNA_AVAILABLE:
        print("\n" + "═"*60)
        print("Optuna Hyperparameter Optimization (50 trials)")
        print("═"*60)

        study = optuna.create_study(direction="maximize",
                                    study_name="ddqn_cartpole")
        study.optimize(lambda t: optuna_objective(t, episodes=300),
                       n_trials=50, show_progress_bar=True)
        

        print("\n  Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"    {k:<20}: {v}")
        print(f"  Best Mean Reward (last 50 ep): {study.best_value:.2f}")

        # Optuna Visualization
        try:
            from optuna.visualization.matplotlib import (
                plot_param_importances, plot_optimization_history)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            plt.sca(axes[0])
            plot_optimization_history(study)
            axes[0].set_title("Optimization History")
            plt.sca(axes[1])
            plot_param_importances(study)
            axes[1].set_title("Parameter Importances")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "optuna_results.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"  [INFO] Optuna grafik hatası: {e}")

        best_cfg = study.best_params
    else:
        #if optuna is not available, we will use a default set of hyperparameters that are known to perform well for DDQN on CartPole-v1. These values are based on common practices in the literature and may not be optimal, but they provide a reasonable starting point for evaluation.
        best_cfg = {"lr": 5e-4, "hidden_dim": 128, "gamma": 0.99,
                    "epsilon_decay": 0.995, "tau": 0.005,
                    "replay_size": 128, "dueling": True}
        print("\n  [INFO] Optuna not available → using default best_cfg.")

    # ══════════════════════════════════════════════════════
    #  PHASE 3 — Multi-seed Statistical Evaluation
    # ══════════════════════════════════════════════════════
    
    print("\n" + "═"*60)
    print(f"  PHASE 3: Multi-seed Statistical Evaluation ({SEEDS})")
    print("═"*60)

    multi_cfg = {
        "use_replay"     : True,
        "use_soft_update": True,
        "tau"            : best_cfg.get("tau", 0.005),
        "gamma"          : best_cfg.get("gamma", 0.99),
        "epsilon_decay"  : best_cfg.get("epsilon_decay", 0.995),
        "replay_size"    : best_cfg.get("replay_size", 128),
        "hidden_dim"     : best_cfg.get("hidden_dim", 128),
        "lr"             : best_cfg.get("lr", 5e-4),
        "dueling"        : best_cfg.get("dueling", True),
        "title"          : "DDQN Best Config",
    }

    print(f"\nBest Config: {multi_cfg}")
    stats = multi_seed_eval(multi_cfg, seeds=SEEDS,
                            episodes=EPISODES, env_name=ENV_NAME)

    print(f"\nFinal Performance (son 50 ep, {len(SEEDS)} seed):")
    print(f"Mean ± Std : {stats['final_mean']:.2f} ± {stats['final_std']:.2f}")

    #───────────────────Mean ± Std Graphic───────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x    = np.arange(len(stats["mean"]))
    smooth_w = 20

    def sm(a): return np.convolve(a, np.ones(smooth_w)/smooth_w, mode='valid')

    m_s  = sm(stats["mean"])
    std_s= sm(stats["std"])
    x_s  = x[:len(m_s)]

    ax.plot(x_s, m_s, color='steelblue', lw=2, label='Mean Reward')
    ax.fill_between(x_s, m_s - std_s, m_s + std_s,
                    alpha=0.25, color='steelblue', label='±1 Std')
    ax.axhline(475, c='red', ls='--', lw=1.2, label='Goal (475)')
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.set_title(f"DDQN Best Config — {len(SEEDS)} Seeds Mean ± Std",
                 fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ddqn_multiseed.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n[INFO] Outputs saved: {OUTPUT_DIR}")
    print("  ddqn_benchmark.png  — Config Comparison")
    print("  optuna_results.png  — Importance of Hyperparameters (Optuna)")
    print("  ddqn_multiseed.png  — Mean ± Std")
    