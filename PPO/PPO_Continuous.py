# ============================================================
#  PPO — Optimized for Academic Benchmark
#  Environment : Pendulum-v1  (Continuous Action Space)
#  Framework   : PyTorch
# ============================================================

import copy
import os
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("pip install optuna")

# ============================================================
# SEED CONTROL 
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# ACTOR  (stochastic policy → mean + sigma)
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_bound: float):
        super().__init__()
        self.action_bound = action_bound

        self.common = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.mu_head    = nn.Linear(128, 1)
        self.sigma_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        feat  = self.common(x)
        mean  = torch.tanh(self.mu_head(feat)) * self.action_bound
        sigma = torch.clamp(torch.nn.functional.softplus(self.sigma_head(feat)), min=1e-3, max=2.0)
        return mean, sigma

# ============================================================
#  CRITIC  (state value V(s))
# ============================================================

class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================
#  PPO AGENT
# ============================================================

class PPOAgent:
    def __init__(self, state_dim: int, action_bound: float,
                 lr_actor: float  = 1e-4,
                 lr_critic: float = 1e-4,
                 epsilon: float   = 0.2,
                 update_epochs: int = 5,
                 gamma: float     = 0.99,
                 lam: float       = 0.95):

        self.epsilon       = epsilon
        self.update_epochs = update_epochs
        self.gamma         = gamma
        self.lam           = lam

        self.pi    = Actor(state_dim, action_bound)
        self.piold = Actor(state_dim, action_bound)
        self.v     = Critic(state_dim)

        self._hard_update(self.pi, self.piold)
        self.piold.eval()

        self.a_opt   = torch.optim.Adam(self.pi.parameters(), lr=lr_actor)
        self.c_opt   = torch.optim.Adam(self.v.parameters(), lr=lr_critic)
        self.c_loss  = nn.MSELoss()

    # ───────────── Action sampling ─────────────
    def get_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs)
        mean, sigma = self.pi(obs_t)
        dist     = Normal(mean.detach(), sigma.detach())
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy(), log_prob

    # ───────────── GAE ─────────────
    def compute_gae(self, next_state: np.ndarray,
                    values: list, rewards: list, masks: list):

        returns    = []
        gae        = 0
        next_val   = self.v(torch.FloatTensor(next_state)).detach()
        values_ext = values + [next_val]

        for step in reversed(range(len(rewards))):
            delta = (rewards[step]
                     + self.gamma * values_ext[step + 1] * masks[step]
                     - values_ext[step])
            gae   = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values_ext[step])
        return returns

    # ───────────── Advantage ─────────────
    def compute_advantage(self, returns: list, states: torch.Tensor):
        ret = torch.cat(returns).detach()
        val = self.v(states).detach()
        adv = ret - val
        # Normalization: zero mean, unit std → gradient stability
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv

    # ───────────── Critic update ─────────────
    def update_critic(self, states: torch.Tensor, returns: list):
        ret = torch.cat(returns).detach()
        val = self.v(states)
        loss = (ret - val).pow(2).mean()
        self.c_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=0.5)
        self.c_opt.step()
        return loss.item()
    
    # ───────────── Actor update (PPO-Clip) ─────────────
    def update_actor(self, states: torch.Tensor,
                     actions: torch.Tensor,
                     advantage: torch.Tensor,
                     mini_batch_size: int):
        batch_size = states.size(0)
        actual_mb_size = min(mini_batch_size, batch_size)
        rand_ids   = np.random.choice(batch_size, actual_mb_size, replace=False)

        s_b   = states[rand_ids]
        a_b   = actions[rand_ids]
        adv_b = advantage[rand_ids]

        mean_new,  sigma_new  = self.pi(s_b)
        with torch.no_grad():
            mean_old,  sigma_old  = self.piold(s_b)

        dist_new = Normal(mean_new, sigma_new)
        dist_old = Normal(mean_old, sigma_old)

        logp_new = dist_new.log_prob(a_b.reshape(-1, 1))
        logp_old = dist_old.log_prob(a_b.reshape(-1, 1))

        ratio = torch.exp(logp_new - logp_old)  
        surrogate1 = ratio * adv_b
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_b
        actor_loss = -torch.mean(torch.min(torch.cat((surrogate1, surrogate2), dim=1), dim=1).values)
        entropy = dist_new.entropy().mean()
        actor_loss = actor_loss - 0.01 * entropy

        self.a_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=0.5)
        self.a_opt.step()
        return actor_loss.item()

    # ───────────── Full update ─────────────
    def update(self, states_np, actions_np, returns,
               mini_batch_size: int):
        self._hard_update(self.pi, self.piold)

        states  = torch.FloatTensor(states_np)
        actions = torch.FloatTensor(actions_np)
        adv     = self.compute_advantage(returns, states)

        a_losses, c_losses = [], []
        for _ in range(self.update_epochs):
            a_losses.append(
                self.update_actor(states, actions, adv, mini_batch_size))
            c_losses.append(
                self.update_critic(states, returns))

        return np.mean(a_losses), np.mean(c_losses)

    # ───────────── Hard update (pi → piold) ─────────────
    @staticmethod
    def _hard_update(src, tgt):
        tgt.load_state_dict(copy.deepcopy(src.state_dict()))

# ============================================================
#  TRAINING LOOP
# ============================================================

def train_ppo(env, agent: PPOAgent,
              epochs: int,
              max_steps: int       = 3200,
              t_len: int           = 64,
              mini_batch_size: int = 32,
              reward_scale: bool   = True,
              early_stop_reward: float  = 3000.0,
              early_stop_window: int    = 20,
              verbose: bool        = True) -> dict:

    ep_rewards, actor_losses, critic_losses = [], [], []

    for epoch in range(1, epochs + 1):
        obs, _ = env.reset()                             
        state_dim = env.observation_space.shape[0]
        obs = obs.reshape(1, state_dim)                  

        ep_reward = 0.0
        values, masks, rewards_buf = [], [], []
        memory_s, memory_a = [], []
        ep_a_losses, ep_c_losses = [], []

        for t in range(max_steps):
            action, _ = agent.get_action(obs)
            value      = agent.v(torch.FloatTensor(obs))

            obs_next, reward, terminated, truncated, _ = env.step(
                action.reshape(1, 1))
            done = terminated or truncated
            obs_next = obs_next.reshape(1, state_dim)

            if reward_scale:
                reward = float(np.squeeze(reward + 16) / 16)            # original reward [-16, 0] → scaled reward [0, 1]
            else:

                reward = float(np.squeeze(reward))

            memory_s.append(obs.squeeze())
            memory_a.append(action.item())
            rewards_buf.append(reward)
            values.append(value)
            # FIX 1: masks — done=True → 0.0 (next step doesn't exist), done=False → 1.0 (next step exists)
            masks.append(0.0 if done else 1.0)

            ep_reward += reward
            obs = obs_next

            update_now = ((t + 1) % t_len == 0) or (t == max_steps - 1)
            if update_now and len(memory_s) > 0:
                s_arr = np.array(memory_s)
                a_arr = np.array(memory_a)

                returns  = agent.compute_gae(
                    obs_next, values, rewards_buf, masks)
                a_loss, c_loss = agent.update(
                    s_arr, a_arr, returns, mini_batch_size)

                ep_a_losses.append(a_loss)
                ep_c_losses.append(c_loss)

                # Buffer cleaning after update
                memory_s, memory_a = [], []
                rewards_buf, values, masks = [], [], []

            if done:
                obs, _ = env.reset()
                obs = obs.reshape(1, state_dim)

        ep_rewards.append(ep_reward)
        actor_losses.append(np.mean(ep_a_losses)  if ep_a_losses  else 0.0)
        critic_losses.append(np.mean(ep_c_losses) if ep_c_losses else 0.0)

        if verbose and epoch % 10 == 0:
            avg = np.mean(ep_rewards[-early_stop_window:])
            print(f"  Ep {epoch:4d} | Reward: {ep_reward:8.2f} | "
                  f"Avg{early_stop_window}: {avg:8.2f} | "
                  f"A_loss: {actor_losses[-1]:7.4f} | "
                  f"C_loss: {critic_losses[-1]:7.4f}")

        # Early stopping
        if (len(ep_rewards) >= early_stop_window and
                np.mean(ep_rewards[-early_stop_window:]) >= early_stop_reward):
            if verbose:
                print(f"\n  ✓ Early stop at epoch {epoch} "
                      f"(avg{early_stop_window} = "
                      f"{np.mean(ep_rewards[-early_stop_window:]):.2f})")
            break

    return {"rewards"       : ep_rewards,
            "actor_losses"  : actor_losses,
            "critic_losses" : critic_losses}

# ============================================================
#  Academic-style Plotting
# ============================================================

def plot_academic(all_results: dict, smooth_window: int = 10,
                  goal: float = 3000.0, save_path: str = None):
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(all_results)))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid') if len(x) >= w else x

    for color, (label, res) in zip(colors, all_results.items()):
        r  = np.array(res["rewards"])
        al = np.array(res["actor_losses"])
        cl = np.array(res["critic_losses"])

        axes[0].plot(smooth(r, smooth_window),  label=label, color=color, lw=1.6)
        axes[1].hist(r[-50:], bins=20, alpha=0.55, label=label, color=color)
        axes[2].plot(smooth(al, smooth_window), color=color, lw=1.4, label=label)
        axes[3].plot(smooth(cl, smooth_window), color=color, lw=1.4, label=label)

    axes[0].axhline(goal, c='red', ls='--', lw=1.2, label=f'Goal ({goal})')
    axes[1].axvline(goal, c='red', ls='--', lw=1.2, label=f'Goal ({goal})')

    titles  = ["Smoothed Reward", "Reward Distribution (Last 50 ep)",
               "Actor Loss", "Critic Loss"]
    xlabels = ["Episode", "Reward", "Episode", "Episode"]
    ylabels = ["Reward (scaled)", "Frequency", "Actor Loss", "Critic Loss"]

    for ax, t, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_title(t, fontsize=11, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("PPO — Pendulum-v1 Benchmark",
                 fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphics saved: {save_path}")
    plt.show()

# ============================================================
# Statistical Evaluation (Multi-seed) 
# ============================================================

def multi_seed_eval(config: dict, seeds: list, epochs: int,
                    env_name: str = "Pendulum-v1") -> dict:
    all_rewards = []

    for seed in seeds:
        set_seed(seed)
        env = gym.make(env_name)
        env.reset(seed=seed)
        agent = PPOAgent(
            state_dim   = env.observation_space.shape[0],
            action_bound= float(env.action_space.high[0]),
            lr_actor    = config.get("lr_actor",  1e-4),
            lr_critic   = config.get("lr_critic", 1e-4),
            epsilon     = config.get("epsilon",   0.2),
            update_epochs= config.get("update_epochs", 5),
            gamma       = config.get("gamma",     0.99),
            lam         = config.get("lam",       0.95),
        )
        res = train_ppo(env, agent,
                        epochs       = epochs,
                        max_steps    = config.get("max_steps",    3200),
                        t_len        = config.get("t_len",        64),
                        mini_batch_size = config.get("mini_batch_size", 32),
                        verbose      = False)
        all_rewards.append(res["rewards"])
        env.close()
        print(f"    Seed {seed} → Mean reward: {np.mean(res['rewards']):.2f}")

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

def optuna_objective(trial, env_name="Pendulum-v1",
                     epochs=300, seed=42):
    set_seed(seed)

    lr_actor      = trial.suggest_float("lr_actor",    1e-4, 1e-3, log=True)
    lr_critic     = trial.suggest_float("lr_critic",   1e-4, 1e-3, log=True)
    epsilon       = trial.suggest_float("epsilon",     0.1,  0.3)
    gamma         = trial.suggest_float("gamma",       0.95, 0.999)
    lam           = trial.suggest_float("lam",         0.90, 0.99)
    update_epochs = trial.suggest_int  ("update_epochs", 3,  10)
    t_len         = trial.suggest_categorical("t_len", [32, 64, 128])
    mini_batch    = trial.suggest_categorical("mini_batch_size", [16, 32, 64])
    
    if mini_batch > t_len:
        raise optuna.exceptions.TrialPruned()

    env = gym.make(env_name)
    env.reset(seed=seed)
    agent = PPOAgent(
        state_dim    = env.observation_space.shape[0],
        action_bound = float(env.action_space.high[0]),
        lr_actor     = lr_actor,
        lr_critic    = lr_critic,
        epsilon      = epsilon,
        update_epochs= update_epochs,
        gamma        = gamma,
        lam          = lam,
    )
    res = train_ppo(env, agent, epochs=epochs,
                    t_len=t_len, mini_batch_size=mini_batch,
                    verbose=False)
    env.close()
    return np.mean(res["rewards"][-20:])

# ============================================================
#  MAIN  —  3 Step Evaluation Pipeline
# ============================================================

if __name__ == "__main__":

    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Graphics will be saved to: {OUTPUT_DIR}")

    ENV_NAME = "Pendulum-v1"
    EPOCHS   = 500
    SEEDS    = [0, 1, 2, 3, 4]

    #Configurations — T_len and mini_batch effects
    configs = [
        dict(t_len=64,  mini_batch_size=32, title="PPO | T=64,  MB=32"),
        dict(t_len=128, mini_batch_size=64, title="PPO | T=128, MB=64"),
    ]

    # ══════════════════════════════════════════════════════
    # Phase 1 — Configuration Comparison (seed=42
    # ══════════════════════════════════════════════════════

    print("\n" + "═"*60)
    print("  Phase 1: Configuration Comparison (seed=42)")
    print("═"*60)

    set_seed(42)
    all_results = {}

    for cfg in configs:
        print(f"\n  ▶ {cfg['title']}")
        env   = gym.make(ENV_NAME)
        env.reset(seed=42)
        agent = PPOAgent(
            state_dim    = env.observation_space.shape[0],
            action_bound = float(env.action_space.high[0]),
        )
        res = train_ppo(env, agent, epochs=EPOCHS,
                        t_len           = cfg["t_len"],
                        mini_batch_size = cfg["mini_batch_size"],
                        verbose         = True)
        all_results[cfg["title"]] = res
        env.close()

    plot_academic(all_results,
                  save_path=os.path.join(OUTPUT_DIR, "ppo_benchmark.png"))

    print("\n" + "─"*60)
    print(f"{'Configuration':<30} {'Mean':>10} {'Max':>10} {'Episodes':>9}")
    print("─"*60)

    for label, res in all_results.items():
        r = res["rewards"]
        print(f"{label:<28} {np.mean(r):>10.2f} "
              f"{np.max(r):>10.2f} {len(r):>9d}")
    print("─"*60)

    # ══════════════════════════════════════════════════════
    #  Phase 2 — Optuna
    # ══════════════════════════════════════════════════════

    if OPTUNA_AVAILABLE:
        print("\n" + "═"*60)
        print("  Phase 2: Optuna Hyperparameter Optimization (50 trial)")
        print("═"*60)

        STUDY_DB   = os.path.join(OUTPUT_DIR, "ppo_optuna.db")
        STUDY_NAME = "ppo_pendulum"
        storage    = f"sqlite:///{STUDY_DB}"

        study = optuna.create_study(direction="maximize",
                                    study_name=STUDY_NAME,
                                    storage=storage,
                                    load_if_exists=True)
        remaining = 50 - len(study.trials)
        if remaining > 0:
            print(f"{remaining} trials are being executed...")
            study.optimize(lambda t: optuna_objective(t, epochs=300),
                        n_trials=remaining, show_progress_bar=True, catch = (ValueError, RuntimeError))
        else:
            print(f"  [INFO] Study is already completed ({len(study.trials)} trial)")

        print("\n  Best Hyperparameters:")

        for k, v in study.best_params.items():
            print(f"    {k:<20}: {v}")
        print(f"  ✓ Best Mean Reward (last 20 episodes): {study.best_value:.2f}")

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
            plt.savefig(os.path.join(OUTPUT_DIR, "ppo_optuna_results.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"  [INFO] Optuna graph error: {e}")

        best_cfg = study.best_params
    else:
        best_cfg = {"lr_actor": 1e-4, "lr_critic": 1e-4, "epsilon": 0.2,
                    "gamma": 0.99, "lam": 0.95, "update_epochs": 5,
                    "t_len": 64, "mini_batch_size": 32}
        print("\n  [INFO] Optuna not available → using default best_cfg.")

    # ══════════════════════════════════════════════════════
    #  Phase 3 — Multiple Seeds
    # ══════════════════════════════════════════════════════

    print("\n" + "═"*60)
    print(f"  Phase 3: Multiple Seed Evaluation ({SEEDS})")
    print("═"*60)

    multi_cfg = {
        "lr_actor"      : best_cfg.get("lr_actor",       1e-4),
        "lr_critic"     : best_cfg.get("lr_critic",      1e-4),
        "epsilon"       : best_cfg.get("epsilon",         0.2),
        "gamma"         : best_cfg.get("gamma",           0.99),
        "lam"           : best_cfg.get("lam",             0.95),
        "update_epochs" : best_cfg.get("update_epochs",    5),
        "t_len"         : best_cfg.get("t_len",            64),
        "mini_batch_size": best_cfg.get("mini_batch_size", 32),
    }

    print(f"\n  ▶ Best Config: {multi_cfg}")
    stats = multi_seed_eval(multi_cfg, seeds=SEEDS,
                            epochs=EPOCHS, env_name=ENV_NAME)

    print(f"\n  ✓ Final Performance (last 20 ep, {len(SEEDS)} seed):")
    print(f"    Mean ± Std : {stats['final_mean']:.2f} ± {stats['final_std']:.2f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    smooth_w = 10
    def sm(a): return np.convolve(a, np.ones(smooth_w)/smooth_w, mode='valid')
    m_s   = sm(stats["mean"])
    std_s = sm(stats["std"])
    x_s   = np.arange(len(m_s))

    ax.plot(x_s, m_s, color='mediumpurple', lw=2, label='Mean Reward')
    ax.fill_between(x_s, m_s - std_s, m_s + std_s,
                    alpha=0.25, color='mediumpurple', label='±1 Std')
    ax.axhline(3000, c='red', ls='--', lw=1.2, label='Goal (3000)')
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward (scaled)")
    ax.set_title(f"PPO Best Config — {len(SEEDS)} Seeds Mean ± Std",
                 fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ppo_multiseed.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n All of the outputs saved to: {OUTPUT_DIR}")
    print("  ppo_benchmark.png       — Config Comparison")
    print("  ppo_optuna_results.png  — Hyperparameter Importance (Optuna)")
    print("  ppo_multiseed.png       — Mean ± Std (Statistical Reliability)")