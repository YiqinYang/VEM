# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(16)

plt.rcParams['pdf.use14corefonts'] = True
plt.rc('text', usetex=True)

GAMMA = 0.9

class RandomMdp():
    def __init__(self, random=False):
        self.N_S = 5
        self.N_A = 3
        self.initial_state_dist = np.ones(self.N_S)
        self.initial_state_dist /= self.initial_state_dist.sum() # uniform reset
        if random:
            # random env
            self.P = np.random.rand(self.N_S, self.N_A, self.N_S)
            self.P /= self.P.sum(-1, keepdims=True)
        else:
            self.P = np.random.rand(self.N_S, self.N_A, self.N_S)
            self.P[self.P == self.P.max(-1, keepdims=True)] = 1
            self.P = np.floor(self.P)
            assert np.all(self.P.sum(-1) == 1), self.P
            # deterministic env
        self.r = np.random.rand(self.N_S, self.N_A)  # r(s, a)
        self.max_steps = 100
    
    def reset(self):
        self.t = 0
        self.state = np.random.choice(self.N_S, p=self.initial_state_dist)
        return self.state
    
    def step(self, action):
        self.t += 1
        reward = self.r[self.state, action]
        self.state = np.random.choice(self.N_S, p=self.P[self.state, action])
        return self.state, reward, self.t >= self.max_steps
    
class RandomPolicy():
    def __init__(self):
        self.N_S = 5
        self.N_A = 3
        self.policy = np.random.rand(self.N_S, self.N_A)
        self.policy /= self.policy.sum(-1, keepdims=True)
    
    def sample(self, state):
        return np.random.choice(self.N_A, p=self.policy[state])
    
    def set_policy(self, mu):
        self.policy = mu

def generate_traj(mdp, policy):
    traj = []
    state = mdp.reset()
    done = False
    while not done:
        action = policy.sample(state)
        next_state, reward, done = mdp.step(action)
        traj.append((state, action, reward, next_state))
        state = next_state
    return traj

def get_fixed_point(operator, Q, mdp, *args, eps=1e-6):
    Q_old = - 1 / (1 - GAMMA) # arbitray init
    while np.max(np.abs(Q_old - Q)) > eps:
        Q_old = Q
        Q = operator(Q, mdp, *args)
    return Q

# --------------------------------Q Based------------------------------------
def optimal_operator(Q, mdp, *args):
    """
    Optimal Bellman Operator: Q(s, a) = r(s, a) + gamma * max_{a'} Q(s', a')
    """
    return mdp.r + GAMMA * np.dot(mdp.P, Q.max(-1))

def eval_operator(Q, mdp, mu, *args):
    """
    Bellman Operator: Q(s, a) = r(s, a) + γ * E_μ Q(s', a')
    """
    return mdp.r + GAMMA * np.dot(mdp.P, np.sum(mu * Q, axis=-1))

def nstep_operator(Q, mdp, mu, nstep=10, *args):
    """
    Nstep Bellman Operator: Q(s, a) = E_μ [\sum_{t=0}^{n-1} γ^t r_t + γ^n * max_{a"} Q(s", a")]
    """
    assert nstep >= 1
    Q = optimal_operator(Q, mdp)
    for _ in range(nstep - 1):
        Q = eval_operator(Q, mdp, mu)
    return Q

def gem_operator(Q, mdp, mu, nstep=10, *args):
    """
    Gem Bellman Operator
    """
    Q = optimal_operator(Q, mdp)
    Qs = [Q]
    for _ in range(nstep - 1):
        Q = eval_operator(Q, mdp, mu)
        Qs.append(Q)
    return np.max(Qs, axis=0)

def T_hat_optimal_operator(traj, Q, *arg):
    state, action, reward, next_state = traj[0]
    return reward + GAMMA * np.max(Q[next_state])

def T_hat_nstep_operator(traj, Q, nstep=10, *arg):
    ret = GAMMA**nstep * np.max(Q[traj[nstep-1][-1]])
    for i, sample in enumerate(traj[:nstep]):
        state, action, reward, next_state = sample
        ret += GAMMA**i * reward
    return ret

def T_hat_gem_operator(traj, Q, nstep=10, *arg):
#     ret = GAMMA**nstep * np.max(Q[traj[nstep-1][-1]])
    rew_n = 0
    Rs = []
    for i, sample in enumerate(traj[:nstep]):
        state, action, reward, next_state = sample
        rew_n += GAMMA**i * reward
        Rs.append(rew_n + GAMMA**(i+1) * np.max(Q[next_state]))
    return np.max(Rs)

def test_optimal_opertor(mdp, mu, Q):
#     Q = np.random.rand(mdp.N_S, mdp.N_A) / (1 - GAMMA)
    Q = np.zeros((mdp.N_S, mdp.N_A))
#     Q = np.ones((mdp.N_S, mdp.N_A)) / (1 - GAMMA)
    T_tilde_Q = optimal_operator(Q, mdp)
    Q_tilde = get_fixed_point(optimal_operator, Q, mdp)
    Q_star = get_fixed_point(optimal_operator, Q, mdp)
    bias = np.max(np.abs(Q_star - Q_tilde))
    contraction = np.max(np.abs(T_tilde_Q - Q_tilde)) / np.max(np.abs(Q - Q_tilde))
    samples = []
    for _ in range(1000):
        traj = generate_traj(mdp, mu)
        state, action, _, _ = traj[0]
        T_hat_Q = T_hat_optimal_operator(traj, Q)
        samples.append(T_hat_Q - T_tilde_Q[state, action])
    variance = np.power(samples, 2).mean()
    return contraction, variance, bias

def test_nstep_opertor(mdp, mu, Q, nstep=10):
    T_tilde_Q = nstep_operator(Q, mdp, mu.policy, nstep)
    Q_tilde = get_fixed_point(nstep_operator, Q, mdp, mu.policy, nstep)
    Q_star = get_fixed_point(optimal_operator, Q, mdp)
    bias = np.max(np.abs(Q_star - Q_tilde))
    contraction = np.max(np.abs(T_tilde_Q - Q_tilde)) / np.max(np.abs(Q - Q_tilde))
    span_bias = np.max(Q_star - Q_tilde) - np.min(Q_star - Q_tilde)
    samples = []
    for _ in range(1000):
        traj = generate_traj(mdp, mu)
        state, action, _, _ = traj[0]
        T_hat_Q = T_hat_nstep_operator(traj, Q, nstep)
        samples.append(T_hat_Q - T_tilde_Q[state, action])
    variance = np.power(samples, 2).mean()
    return contraction, variance, bias, span_bias

def test_gem_opertor(mdp, mu, Q, nstep=10):
    T_tilde_Q = gem_operator(Q, mdp, mu.policy, nstep)
    Q_tilde = get_fixed_point(gem_operator, Q, mdp, mu.policy, nstep)
    Q_star = get_fixed_point(optimal_operator, Q, mdp)
    bias = np.max(np.abs(Q_star - Q_tilde))
    contraction = np.max(np.abs(T_tilde_Q - Q_tilde)) / np.max(np.abs(Q - Q_tilde))
    samples = []
    for _ in range(1000):
        traj = generate_traj(mdp, mu)
        state, action, _, _ = traj[0]
        T_hat_Q = T_hat_gem_operator(traj, Q, nstep)
        samples.append(T_hat_Q - T_tilde_Q[state, action])
    variance = np.power(samples, 2).mean()
    return contraction, variance, bias

# --------------------------------V Based------------------------------------
def evl_operator(V, mdp, mu, tau, *args):
    """
    Expectile Bellman Operator: V(s) = E_μ [V(s) + (r(s, a) + gamma * V(s') - V(s))_+ + (r(s, a) + gamma * V(s') - V(s))_-]
    """
    A_u = np.maximum(mdp.r + GAMMA * np.dot(mdp.P, V) - V[:, np.newaxis], 0)
    A_d = np.minimum(mdp.r + GAMMA * np.dot(mdp.P, V) - V[:, np.newaxis], 0)
    A = tau * A_u + (1 - tau) * A_d
    alpha = 1 / max(tau, 1 - tau)
    V = V + alpha * np.sum(mu * A, axis=1)
    return V

def T_hat_evl_operator(traj, V, tau, *arg):
    state, action, reward, next_state = traj[0]
    A_u = np.maximum(reward + GAMMA * V[next_state] - V[state], 0)
    A_d = np.minimum(reward + GAMMA * V[next_state] - V[state], 0)
    A = tau * A_u + (1 - tau) * A_d
    alpha = 1 / max(tau, 1 - tau)
    return V[state] + alpha * A 

def test_evl_opertor(mdp, mu, V, tau=0.9):
    T_tilde_V = evl_operator(V, mdp, mu.policy, tau)
    V_tilde = get_fixed_point(evl_operator, V, mdp, mu.policy, tau)
    V_star = np.max(get_fixed_point(optimal_operator, np.zeros((mdp.N_S, mdp.N_A)), mdp), axis=1)
    bias = np.max(np.abs(V_star - V_tilde))
    span_bias = np.max(V_star - V_tilde) - np.min(V_star - V_tilde)
    contraction = np.max(np.abs(T_tilde_V - V_tilde)) / np.max(np.abs(V - V_tilde))
    samples = []
    for _ in range(1000):
        traj = generate_traj(mdp, mu)
        state, action, _, _ = traj[0]
        T_hat_V = T_hat_evl_operator(traj, V, tau)
        samples.append(T_hat_V - T_tilde_V[state])
    variance = np.power(samples, 2).mean()
    return contraction, variance, bias, span_bias

def eval_v_operator(V, mdp, mu, *args):
    """
    Bellman Operator: V(s) = E_μ [r(s, a) + γ * V(s')]
    """
    return np.sum(mu * (mdp.r + GAMMA * np.dot(mdp.P, V)), axis=-1)

def vem_operator(V, mdp, mu, nstep=10, tau=0.9, *args):
    """
    Vem Bellman Operator
    """
    Vs = []
    for _ in range(nstep - 1):
        V = eval_v_operator(V, mdp, mu)
        Vs.append(V)
    V = evl_operator(V, mdp, mu, tau)
    Vs.append(V)
    return np.max(Vs, axis=0)

def vem2_operator(V, mdp, mu, nstep=10, tau=0.9, *args):
    """
    Vem Bellman Operator
    """
    V = evl_operator(V, mdp, mu, tau)
    Vs = [V]
    for _ in range(nstep - 1):
        V = eval_v_operator(V, mdp, mu)
        Vs.append(V)
    return np.max(Vs, axis=0)

def T_hat_vem_operator(traj, V, nstep=10, tau=0.9, *arg):
    rew_n = 0
    Rs = []
    for i, sample in enumerate(traj[:nstep]):
        state, action, reward, next_state = sample
        rew_n += GAMMA**i * reward
        Rs.append(rew_n + GAMMA**(i+1) * V[next_state])
    V_target = np.max(Rs)
    state, action, reward, next_state = traj[0]
    V_cur = V[state]
    A_u = np.maximum(V_target - V_cur, 0)
    A_d = np.minimum(V_target - V_cur, 0)
    A = tau * A_u + (1 - tau) * A_d
    alpha = 1 / max(tau, 1 - tau)
    return V_cur + alpha * A

def T_hat_vem2_operator(traj, V, nstep=10, tau=0.9, *arg):
    rew_n = 0
    Rs = []
    for i, sample in enumerate(traj[:nstep]):
        state, action, reward, next_state = sample
        A_u = np.maximum(reward + GAMMA * V[next_state] - V[state], 0)
        A_d = np.minimum(reward + GAMMA * V[next_state] - V[state], 0)
        A = tau * A_u + (1 - tau) * A_d
        alpha = 1 / max(tau, 1 - tau)
        V_next = V[state] + alpha * A
        Rs.append(rew_n + GAMMA**i * V_next)
        rew_n += GAMMA**i * reward
    return np.max(Rs)

def test_vem_opertor(mdp, mu, V, nstep=10, tau=0.9):
    T_tilde_V = vem_operator(V, mdp, mu.policy, nstep, tau)
    V_tilde = get_fixed_point(vem_operator, V, mdp, mu.policy, nstep, tau)
    V_star = np.max(get_fixed_point(optimal_operator, np.zeros((mdp.N_S, mdp.N_A)), mdp), axis=1)
    bias = np.max(np.abs(V_star - V_tilde))
    span_bias = np.max(V_star - V_tilde) - np.min(V_star - V_tilde)
    contraction = np.max(np.abs(T_tilde_V - V_tilde)) / np.max(np.abs(V - V_tilde))
    samples = []
    for _ in range(1000):
        traj = generate_traj(mdp, mu)
        state, action, _, _ = traj[0]
        T_hat_V = T_hat_vem_operator(traj, V, nstep, tau)
        samples.append(T_hat_V - T_tilde_V[state])
    variance = np.power(samples, 2).mean()
    return contraction, variance, bias, span_bias

def test_vem2_opertor(mdp, mu, V, nstep=10, tau=0.9):
    T_tilde_V = vem2_operator(V, mdp, mu.policy, nstep, tau)
    V_tilde = get_fixed_point(vem2_operator, V, mdp, mu.policy, nstep, tau)
    V_star = np.max(get_fixed_point(optimal_operator, np.zeros((mdp.N_S, mdp.N_A)), mdp), axis=1)
    
    bias = np.max(np.abs(V_star - V_tilde))
    span_bias = np.max(V_star - V_tilde) - np.min(V_star - V_tilde)
    contraction = np.max(np.abs(T_tilde_V - V_tilde)) / np.max(np.abs(V - V_tilde))
    
    samples = []
    for _ in range(1000):
        traj = generate_traj(mdp, mu)
        state, action, _, _ = traj[0]
        T_hat_V = T_hat_vem2_operator(traj, V, nstep, tau)
        samples.append(T_hat_V - T_tilde_V[state])
    variance = np.power(samples, 2).mean()
    return contraction, variance, bias, span_bias

mdp = RandomMdp()
# ---------------------------------example1-------------------------------------
V = np.zeros((mdp.N_S))
np.set_printoptions(precision=3, suppress=True)

def softmax(x, alpha):
    y = x - x.max(axis=-1, keepdims=True)
    y = np.exp(y / alpha)
    return y / np.sum(y, axis=-1, keepdims=True)

mu = RandomPolicy()
Q = np.zeros((mdp.N_S, mdp.N_A))
Q = get_fixed_point(optimal_operator, Q, mdp)
pi = softmax(Q, 1)
pi[:, 0] = 0
pi /= pi.sum(-1, keepdims=True)
mu.set_policy(pi)

V = np.zeros((mdp.N_S))
from itertools import product
exps2 = []
for nstep, tau in product(np.arange(1, 5), np.arange(0.6, 0.9, 0.1)):
    # nstep, tau: 1, 0.6; 1, 0.7; 1, 0.8; 1, 0.9; 2, 0.6; ..., 4, 0.9
    contraction, variance, bias, span_bias = test_vem2_opertor(mdp, mu, V, nstep, tau)
    exps2.append(dict(nstep=nstep, tau=tau, contraction=contraction, variance=variance, bias=bias, span_bias=span_bias))
    print(nstep, f"{tau:.1f}", (contraction, variance, bias, span_bias))

# VEM2 bias
fig = plt.figure(dpi=300, figsize=(6, 2.5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.subplots_adjust(wspace=0.3)
for exp in exps2:
    color = sns.light_palette(sns.color_palette('cool_r', n_colors=4)[exp['nstep']-1], n_colors=6, input='rgb')[int(exp['tau'] * 10.1) - 4]
    ax1.scatter(exp['bias'], exp['contraction'], color=color, s=exp['variance'] * 800, edgecolors='k', linewidths=0.5)
    ax1.set_xlabel('Bias')
    ax1.set_ylabel("Contraction Rate")
    ax2.scatter(exp['contraction'], exp['variance'], color=color, s=exp['bias'] * 100, edgecolors='k', linewidths=0.5)
    ax2.set_xlabel("Contraction Rate")
    ax2.set_ylabel('Variance')
fig.savefig('sec4_figa.pdf', dpi=300, bbox_inches='tight')

# ---------------------------------------example2---------------------------------------------
V = np.zeros((mdp.N_S))
np.set_printoptions(precision=3, suppress=True)

def softmax(x, alpha):
    y = x - x.max(axis=-1, keepdims=True)
    y = np.exp(y / alpha)
    return y / np.sum(y, axis=-1, keepdims=True)

mu = RandomPolicy()
Q = np.zeros((mdp.N_S, mdp.N_A))
Q = get_fixed_point(optimal_operator, Q, mdp)

from itertools import product
nstep = 10
exps_mu = []
for alpha, tau in product([0.1, 0.3, 1, 3], np.arange(0.6, 0.9, 0.1)):
    mu.set_policy(softmax(Q, alpha))
    contraction, variance, bias, span_bias = test_vem2_opertor(mdp, mu, V, nstep, tau)
    exps_mu.append(dict(nstep=nstep, tau=tau, alpha=alpha, contraction=contraction, variance=variance, bias=bias, span_bias=span_bias))
    print(f"{alpha:.3f}", f"{tau:.1f}", (contraction, variance, bias, span_bias))

# VEM2 bias
fig = plt.figure(dpi=300, figsize=(6, 2.5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.subplots_adjust(wspace=0.3)
color_palette = {0.1: 0, 0.3: 1, 1: 2, 3: 3}
for exp in exps_mu:
    color = sns.light_palette(sns.color_palette('hot_r', n_colors=4)[color_palette[exp['alpha']]], n_colors=6, input='rgb')[int(exp['tau'] * 10.1) - 4]
    ax1.scatter(exp['bias'], exp['contraction'], color=color, s=exp['variance'] * 800, edgecolors='k', linewidths=0.5)
    ax1.set_xlabel('Bias')
    ax1.set_ylabel("Contraction Rate")
    ax2.scatter(exp['contraction'], exp['variance'], color=color, s=exp['bias'] * 100, edgecolors='k', linewidths=0.5)
    ax2.set_xlabel("Contraction Rate")
    ax2.set_ylabel('Variance')
# plt.show()
fig.savefig('sec4_figb.pdf', dpi=300, bbox_inches='tight')