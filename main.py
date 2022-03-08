import numpy as np
from numpy.lib.utils import info
import torch
import gym
import argparse
import os

from torch.nn.functional import threshold
import d4rl
from d4rl.offline_env import OfflineEnv
import utils
import TD3
from episodic_memory_tbp import EpisodicMemoryTBP
from torch.utils.tensorboard import SummaryWriter
import time

def reward2return(rewards, gamma=0.99):
    # covert reward to return
    returns = []
    Rtn = 0
    for r in reversed(rewards):
        Rtn = r + gamma * Rtn
        returns.append(Rtn)
    return list(reversed(returns))

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	avg_reward = 0.
	episode_returns = []
	eval_qs = []
	EVAl_MEAN = 0
	EVAL_ABS = 0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state), True)
			state, reward, done, info = eval_env.step(action)
			avg_reward += reward
			episode_returns.append(reward)
			eval_q = policy.eval_state_action(state, action)
			eval_qs.append(np.mean(eval_q))
		episode_returns = reward2return(episode_returns)
		EVAl_MEAN += np.mean([x - y for x, y in zip(eval_qs, episode_returns)])
		EVAL_ABS += np.mean([abs(x - y) for x, y in zip(eval_qs, episode_returns)])
		eval_qs = []
		episode_returns = []	

	avg_reward /= eval_episodes
	EVAl_MEAN /= eval_episodes
	EVAL_ABS /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} env: {str(env_name)}")
	print("---------------------------------------")
	return avg_reward, EVAl_MEAN, EVAL_ABS

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="pen-human-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=100, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used 25e3
	parser.add_argument("--eval_freq", default=1e2, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=128, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim_ = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim_,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

	buffer_size = len(env.get_dataset()['observations'])
	# -----------------------------parameter----------------------------
	train_freq = 100
	gradient_steps = 200
	slope = 0.3
	max_step = 1000
	policy_beta = 100
	threshold_vem = -1000
	eta = 0.1
	# -----------------------------parameter----------------------------
	num_timesteps = 0
	beta = -1

	memory = EpisodicMemoryTBP(buffer_size, state_dim=1,
								obs_space=env.observation_space,
								action_shape=env.action_space,
								gamma=args.discount,
								max_step=max_step,
								policy=policy,
								eta=eta,
								policy_noise=kwargs["policy_noise"], noise_clip=kwargs["noise_clip"], max_action=kwargs["max_action"])

	# -------------------------------load dataset------------------------------------
	dataset = d4rl.sequence_dataset(env)
	reward_l = []
	for seq in dataset:
		observations, actions, dones, rewards, truly_dones = seq['observations'], seq['actions'], seq[
			'timeouts'], seq['rewards'], seq['terminals']
		if dones[-1] == True:
			truly_dones[-1] = True
		elif truly_dones[-1] == True:
			observations = np.vstack((observations, observations[-1].reshape(1, -1)))
			actions = np.vstack((actions, actions[-1].reshape(1, -1)))
			dones = np.hstack((dones, np.array([True])))
			truly_dones = np.hstack((truly_dones, np.array([True])))
			rewards = np.hstack((rewards, rewards[-1].reshape(-1,)))
		reward_l.append(rewards.sum())
		trajectory = [(obs, action, 0, 0, reward, truly_done, done) for
						obs, action, reward, truly_done, done in
						zip(observations, actions, rewards, truly_dones, dones)]
		memory.update_sequence_with_qs(trajectory)
	print('---------------------- done ----------------------------')
	# --------------------------------------------------------------------------------
	policy_name = 'VEM_softmax'
	writer = SummaryWriter(f"results/{str(policy_name) + '_' + str(args.env) + '_' + str(args.seed)}/")
	torch.set_num_threads(10)
	start_time = time.time()
	# -----------------------------update V------------------------------------
	for t in range(int(args.max_timesteps)):
		if t % train_freq == 0:
			memory.update_memory(0, beta=beta)
			print('update_t: ', t, 'time: ', int((time.time() - start_time)), str(args.env), str(slope), str(max_step), str(policy_beta), str(args.seed))
			start_time = time.time()
			for grad_step in range(gradient_steps):
				return_info = policy.train_critic(memory, args.batch_size, gradient_steps, slope, policy_beta)
		if (t + 1) % args.eval_freq == 0:
			writer.add_scalar('V_max', return_info['Q_max'], t)
			writer.add_scalar('V_min', return_info['Q_min'], t)
			writer.add_scalar('V_mean', return_info['Q_mean'], t)
			writer.add_scalar('R_max', return_info['R_max'], t)
			writer.add_scalar('R_min', return_info['R_min'], t)
			writer.add_scalar('R_mean', return_info['R_mean'], t)
			writer.add_scalar('v_loss', return_info['q_loss'], t)

		if (t + 1) % (args.eval_freq * 5) == 0: # args.eval_freq * 5
			policy.save(f"./pytorch_models/{str(args.env) + '_' + str(args.seed)}")

	# -----------------------------update A------------------------------------
	policy.load(f"./pytorch_models/{str(args.env) + '_' + str(args.seed)}")
	memory.update_memory(0, beta=beta)
	for t in range(int(args.max_timesteps)): 
		if t % train_freq == 0:
			for grad_step in range(gradient_steps):
				return_info = policy.train_actor(memory, args.batch_size, gradient_steps, slope, policy_beta, threshold_vem, args.env)
		if (t + 1) % args.eval_freq == 0:
			eval_reward, eval_mean, eval_abs = eval_policy(policy, args.env, args.seed)
			writer.add_scalar('reward', eval_reward, t)
			writer.add_scalar('V-R_mean', eval_mean, t)
			writer.add_scalar('V-R_abs', eval_abs, t)
			try:
				writer.add_scalar('actor_loss', return_info['actor_loss'], t)
				writer.add_scalar('logp', return_info['logp'], t)
			except:
				pass