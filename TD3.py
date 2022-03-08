import copy
from math import gamma
import numpy as np
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import d4rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class awacMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_logits = nn.Parameter(
                    torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit

        log_std = torch.sigmoid(self.log_std_logits)
        
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(-1)
        else:
            logp_pi = None
        return pi_action, logp_pi

    def get_logprob(self, obs, actions):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(-1)
        return logp_pi


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.l4 = nn.Linear(state_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward_Q(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def forward(self, state):
		s = state
		v1 = F.relu(self.l1(s))
		v1 = F.relu(self.l2(v1))
		v1 = self.l3(v1)

		v2 = F.relu(self.l4(s))
		v2 = F.relu(self.l5(v2))
		v2 = self.l6(v2)
		return v1, v2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):
		obs_dim, act_dim = state_dim, action_dim
		self.actor_stc = awacMLPActor(obs_dim, act_dim, (256,256,256,256), nn.ReLU, max_action).to(device)
		self.actor_target_stc = copy.deepcopy(self.actor_stc)
		self.actor_optimizer = torch.optim.Adam(self.actor_stc.parameters(), lr=3e-4)

		self.critic = Critic(state_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)

		self.critic_2 = Critic(state_dim).to(device)
		self.critic_target_2 = copy.deepcopy(self.critic_2)

		self.critic_parameters = list(self.critic.parameters()) + list(self.critic_2.parameters())
		self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=3e-4)
		
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

	def select_action(self, state, deterministic=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		a, _ = self.actor_stc(state, deterministic, False)
		return a.cpu().data.numpy().flatten()


	def eval_state_action(self, state, action):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		action = torch.FloatTensor(action.reshape(1, -1)).to(device)
		V1, V2 = self.critic(state)
		V3, V4 = self.critic_2(state)
		V = torch.stack([V1, V2, V3, V4]).cpu().mean().detach().numpy()
		return V


	def train_critic(self, memory, batch_size=128, gradient_steps=100, slope=0.1, policy_beta=0.5):
		self.total_it += 1
		num_q = 4

		double_type = "identical"
		num_samples_collection = {"identical": batch_size, "inner": batch_size * num_q // 2,
									"both": batch_size * num_q}
		num_samples = num_samples_collection[double_type]
		batch = memory.sample(num_samples, mix=False)
		batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_returns = torch.FloatTensor(batch['obs0']).to(device), torch.FloatTensor(batch[
			'actions']).to(device), torch.FloatTensor(batch['rewards']).to(device), torch.FloatTensor(batch['obs1']).to(device), torch.FloatTensor(batch['terminals1']).to(device), batch['return']
		if double_type == "identical":
			batch_returns_4 = torch.FloatTensor(np.repeat(batch_returns, num_q // 2, axis=1)).to(device)

		# --------------------V estimation-------------------
		V1, V2 = self.critic(batch_obs)
		V3, V4 = self.critic_2(batch_obs)
		V = torch.stack([V1, V2, V3, V4], dim=1).squeeze(-1)
		diff = batch_returns_4 - V
		quota = torch.FloatTensor(np.zeros((batch_size, 4))).to(device)
		# q-loss:
		pre_q_loss = (slope * torch.max(quota, diff) + (1 - slope) * torch.min(quota, diff)) ** 2
		batch_dones_m = torch.stack([batch_dones, batch_dones, batch_dones, batch_dones], dim=1).squeeze(-1)
		q_loss = (pre_q_loss * (1 - batch_dones_m)).mean()

		# # Optimize the critic
		self.critic_optimizer.zero_grad()
		q_loss.backward()
		self.critic_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_((1 - (1 - self.tau) ** (gradient_steps * 1)) * param.data + (1 - self.tau) ** (gradient_steps * 1) * target_param.data)

		for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
			target_param.data.copy_((1 - (1 - self.tau) ** (gradient_steps * 1)) * param.data + (1 - self.tau) ** (gradient_steps * 1) * target_param.data)

		batch_returns = torch.FloatTensor(batch_returns).to(device)
		Q_max = torch.max(V).cpu().detach().numpy()
		Q_min = torch.min(V).cpu().detach().numpy()
		Q_mean = torch.mean(V).cpu().detach().numpy()
		R_max = torch.max(batch_returns).cpu().detach().numpy()
		R_min = torch.min(batch_returns).cpu().detach().numpy()
		R_mean = torch.mean(batch_returns).cpu().detach().numpy()
		
		dict_return = dict(Q_max=Q_max, Q_min=Q_min, Q_mean=Q_mean, R_max=R_max, 
						R_min=R_min, R_mean=R_mean, q_loss=q_loss.cpu().detach().numpy())
		return dict_return
	

	def train_actor(self, memory, batch_size=128, gradient_steps=100, slope=0.1, policy_beta=0.5, threshold_vem=5, env=None):
		num_q = 4
		double_type = "identical"
		num_samples_collection = {"identical": batch_size, "inner": batch_size * num_q // 2,
									"both": batch_size * num_q}
		num_samples = num_samples_collection[double_type]
		batch = memory.sample(num_samples, mix=False)
		batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_returns = torch.FloatTensor(batch['obs0']).to(device), torch.FloatTensor(batch[
			'actions']).to(device), torch.FloatTensor(batch['rewards']).to(device), torch.FloatTensor(batch['obs1']).to(device), torch.FloatTensor(batch['terminals1']).to(device), batch['return']
		if double_type == "identical":
			batch_returns_4 = torch.FloatTensor(np.repeat(batch_returns, num_q // 2, axis=1)).to(device)
		V1, V2 = self.critic(batch_obs)
		V3, V4 = self.critic_2(batch_obs)
		batch_returns = torch.FloatTensor(batch_returns).to(device)
		R_1, R_2 = batch_returns[:, 0].view(batch_size, 1), batch_returns[:, 1].view(batch_size, 1)
		R_min = torch.min(R_1, R_2)
		V_mean = torch.mean(torch.stack([V1, V2, V3, V4], dim=1).squeeze(-1), dim=1).view(batch_size, 1)
		batch_index = torch.where((R_min - V_mean) > threshold_vem)
		Advantage = (R_min - V_mean)[batch_index]
		policy_logpp = (self.actor_stc.get_logprob(batch_obs, batch_actions)).view(batch_size, 1)[batch_index]
		# ---------------------V_MEAN--------------------------
		# --------------------softmax_policy--------------
		weights = F.softmax(Advantage/policy_beta, dim=0).view(-1)
		# --------------------relu_policy-----------------
		# m = nn.LeakyReLU(policy_beta)
		# Adv = ((R_min - V_mean) / ((R_min - V_mean).std())).view(batch_size, 1)[batch_index]
		# weights = m(Adv).view(-1)
	
		# ------------------------ICQ softmax----------------------
		batch_dones_m = batch_dones[batch_index].squeeze(-1)
		pre_actor_loss = -policy_logpp * len(weights) * weights.detach()
		actor_loss = (pre_actor_loss * (1 - batch_dones_m)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		for param, target_param in zip(self.actor_stc.parameters(), self.actor_target_stc.parameters()):
			target_param.data.copy_((1 - (1 - self.tau) ** (gradient_steps * 1)) * param.data + (1 - self.tau) ** (gradient_steps * 1) * target_param.data)

		logp = torch.mean(policy_logpp).cpu().detach().numpy()
		dict_return = dict(actor_loss=actor_loss.cpu().detach(), logp=logp)
		return dict_return


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_2.state_dict(), filename + "_critic_2")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_target_2 = copy.deepcopy(self.critic_2)