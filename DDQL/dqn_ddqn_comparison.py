# ============================================================
#  DQN vs DDQN — Comparison
#  Environment : CartPole-v1
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
import torch.optim as optim


# ============================================================
#  SEED
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
#  NETWORK
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, dueling=False):
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

    def forward(self, x):
        feat = self.feature(x)
        if self.dueling:
            v   = self.value_stream(feat)
            adv = self.adv_stream(feat)
            return v + adv - adv.mean(dim=1, keepdim=True)
        return self.out(feat)


# ============================================================
#  DQN AGENT  (no target network)
# ============================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3):
        self.device    = torch.device("cpu")
        self.model     = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def predict(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(s).squeeze(0)

    def update(self, states, actions, td_targets):
        s  = torch.FloatTensor(np.array(states)).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        tg = torch.FloatTensor(td_targets).to(self.device)
        q_pred = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss   = self.criterion(q_pred, tg)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def replay(self, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return None
        batch       = random.sample(memory, batch_size)
        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int64)
        next_states = np.array([e[2] for e in batch], dtype=np.float32)
        rewards     = np.array([e[3] for e in batch], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)
        ns_tensor   = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_q = self.model(ns_tensor).max(dim=1).values.cpu().numpy()
        td_targets = rewards + gamma * next_q * (1.0 - dones)
        return self.update(states, actions, td_targets)

    def mean_q(self, memory, n=256):
        if len(memory) < n:
            return 0.0
        sample = random.sample(memory, min(n, len(memory)))
        states = torch.FloatTensor(np.array([e[0] for e in sample])).to(self.device)
        with torch.no_grad():
            return self.model(states).max(dim=1).values.mean().item()


# ============================================================
#  DDQN AGENT  (target network )
# ============================================================
class DDQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 lr=1e-3, dueling=False):
        self.device = torch.device("cpu")
        self.model  = QNetwork(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def predict(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(s).squeeze(0)

    def update(self, states, actions, td_targets):
        s  = torch.FloatTensor(np.array(states)).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        tg = torch.FloatTensor(td_targets).to(self.device)
        q_pred = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss   = self.criterion(q_pred, tg)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def replay(self, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return None
        batch       = random.sample(memory, batch_size)
        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int64)
        next_states = np.array([e[2] for e in batch], dtype=np.float32)
        rewards     = np.array([e[3] for e in batch], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)
        ns_tensor   = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            best_actions  = self.model(ns_tensor).argmax(dim=1)
            next_q        = self.target(ns_tensor)
            next_q_values = next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1).cpu().numpy()
        td_targets = rewards + gamma * next_q_values * (1.0 - dones)
        return self.update(states, actions, td_targets)

    def soft_update(self, tau=0.005):
        for tp, mp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(tau * mp.data + (1.0 - tau) * tp.data)

    def mean_q(self, memory, n=256):
        if len(memory) < n:
            return 0.0
        sample = random.sample(memory, min(n, len(memory)))
        states = torch.FloatTensor(np.array([e[0] for e in sample])).to(self.device)
        with torch.no_grad():
            return self.model(states).max(dim=1).values.mean().item()


# ============================================================
#  TRAINING LOOP  (Common for DQN and DDQN )
# ============================================================
def train(env, agent, episodes,
          gamma=0.99, epsilon=1.0, epsilon_min=0.01,
          epsilon_decay=0.9995, replay_size=64,
          memory_size=10_000, min_memory=1_000,
          tau=0.005, early_stop_reward=475.0,
          early_stop_window=50) -> dict:

    memory  = deque(maxlen=memory_size)
    rewards, losses, mean_qs = [], [], []
    is_ddqn = isinstance(agent, DDQNAgent)

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done, total_reward, ep_losses = False, 0.0, []

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.predict(state).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            memory.append((state, action, next_state, reward, float(terminated)))

            if len(memory) >= min_memory:
                loss = agent.replay(memory, replay_size, gamma)
                if loss is not None:
                    ep_losses.append(loss)

            if is_ddqn:
                agent.soft_update(tau)

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            state = next_state

        rewards.append(total_reward)
        losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        mean_qs.append(agent.mean_q(memory))

        if (len(rewards) >= early_stop_window and
                np.mean(rewards[-early_stop_window:]) >= early_stop_reward):
            break

    return {"rewards": rewards, "losses": losses, "mean_qs": mean_qs}


# ============================================================
#  MULTI-SEED EVAL
# ============================================================
def run_multi_seed(algo: str, cfg: dict, seeds: list,
                   episodes: int, env_name="CartPole-v1") -> dict:
    all_rewards, all_losses, all_qs = [], [], []

    for seed in seeds:
        set_seed(seed)
        env = gym.make(env_name)
        env.reset(seed=seed)

        if algo == "DQN":
            agent = DQNAgent(
                state_dim  = env.observation_space.shape[0],
                action_dim = int(env.action_space.n),
                hidden_dim = cfg["hidden_dim"],
                lr         = cfg["lr"],
            )
        else:  # DDQN
            agent = DDQNAgent(
                state_dim  = env.observation_space.shape[0],
                action_dim = int(env.action_space.n),
                hidden_dim = cfg["hidden_dim"],
                lr         = cfg["lr"],
                dueling    = cfg.get("dueling", False),
            )

        res = train(env, agent, episodes=episodes,
                    gamma         = cfg["gamma"],
                    epsilon_decay = cfg["epsilon_decay"],
                    replay_size   = cfg["replay_size"],
                    tau           = cfg.get("tau", 0.005))
        env.close()

        all_rewards.append(res["rewards"])
        all_losses.append(res["losses"])
        all_qs.append(res["mean_qs"])
        print(f"    [{algo}] Seed {seed} → "
              f"Mean: {np.mean(res['rewards']):.1f} | "
              f"Final50: {np.mean(res['rewards'][-50:]):.1f}")

    # Pad
    def pad(arrs):
        ml = max(len(a) for a in arrs)
        return np.array([a + [a[-1]] * (ml - len(a)) for a in arrs])

    r = pad(all_rewards)
    l = pad(all_losses)
    q = pad(all_qs)

    return {
        "reward_mean": r.mean(axis=0), "reward_std": r.std(axis=0),
        "loss_mean"  : l.mean(axis=0), "loss_std"  : l.std(axis=0),
        "q_mean"     : q.mean(axis=0), "q_std"     : q.std(axis=0),
        "final_mean" : r[:, -50:].mean(),
        "final_std"  : r[:, -50:].std(),
    }

# ============================================================
#  GRAPHING
# ============================================================
def plot_comparison(results: dict, smooth_window=20,
                    goal=475.0, save_path=None):

    COLORS = {"DQN": "#E55934", "DDQN": "#1B998B"}

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    def sm(x):
        if len(x) >= smooth_window:
            return np.convolve(x, np.ones(smooth_window) / smooth_window, mode='valid')
        return x

    for algo, res in results.items():
        c  = COLORS[algo]
        rm = sm(res["reward_mean"])
        rs = sm(res["reward_std"])
        lm = sm(res["loss_mean"])
        qm = sm(res["q_mean"])
        x  = np.arange(len(rm))

        # Panel 1: Reward
        axes[0].plot(x, rm, color=c, lw=2, label=algo)
        axes[0].fill_between(x, rm - rs, rm + rs, alpha=0.18, color=c)

        # Panel 2: Histogram (son 100 ep)
        last100 = res["reward_mean"][-100:]
        axes[1].hist(last100, bins=20, alpha=0.55, color=c, label=algo)

        # Panel 3: Loss
        axes[2].plot(np.arange(len(lm)), lm, color=c, lw=1.6, label=algo)

        # Panel 4: Mean-Q
        axes[3].plot(np.arange(len(qm)), qm, color=c, lw=1.6, label=algo)

    axes[0].axhline(goal, c='black', ls='--', lw=1.2, label=f'Goal ({goal})')
    axes[1].axvline(goal, c='black', ls='--', lw=1.2, label=f'Goal ({goal})')

    titles  = ["Smoothed Reward (Mean ± Std)",
               "Reward Distribution (Last 100 ep)",
               "Training Loss (Mean)",
               "Mean Q-Value (Mean)"]
    xlabels = ["Episode", "Reward", "Episode", "Episode"]
    ylabels = ["Reward", "Frequency", "Loss (Huber)", "Mean Q"]

    for ax, t, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_title(t, fontsize=11, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("DQN vs Double DQN — CartPole-v1 (5 Seeds)",
                 fontsize=14, fontweight='bold', y=1.01)
    

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [✓] Grafik kaydedildi: {save_path}")
    plt.show()


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    EPISODES   = 500
    SEEDS      = [0, 1, 2, 3, 4]
    ENV_NAME   = "CartPole-v1"

    # ── Best Configs (Optuna) ───────────────────
    DQN_CFG = {
        "lr"            : 0.00318809433913564,
        "hidden_dim"    : 128,
        "gamma"         : 0.9109550047363371,
        "epsilon_decay" : 0.9994952111203538,
        "replay_size"   : 32,
    }

    DDQN_CFG = {
        "lr"            : 0.000266664267814904,
        "hidden_dim"    : 256,
        "gamma"         : 0.984460321578167,
        "epsilon_decay" : 0.9939139245560618,
        "tau"           : 0.011549026245977027,
        "replay_size"   : 64,
        "dueling"       : True,
    }

    results = {}

    print("\n" + "═"*60)
    print("  DQN — 5 Seed")
    print("═"*60)
    results["DQN"] = run_multi_seed("DQN", DQN_CFG, SEEDS, EPISODES, ENV_NAME)

    print("\n" + "═"*60)
    print("  DDQN — 5 Seed")
    print("═"*60)
    results["DDQN"] = run_multi_seed("DDQN", DDQN_CFG, SEEDS, EPISODES, ENV_NAME)

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "─"*55)
    print(f"  {'Algoritma':<10} {'Final Mean':>12} {'Final Std':>10}")
    print("─"*55)
    for algo, res in results.items():
        print(f"  {algo:<10} {res['final_mean']:>12.2f} {res['final_std']:>10.2f}")
    print("─"*55)

    # ── Graph ──────────────────────────────────────────────
    plot_comparison(results,
                    save_path=os.path.join(OUTPUT_DIR, "dqn_vs_ddqn.png"))

    print(f"\n  [✓] dqn_vs_ddqn.png saved: {OUTPUT_DIR}")