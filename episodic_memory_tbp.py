from episodic_memory import EpisodicMemory
import numpy as np
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

class EpisodicMemoryTBP(EpisodicMemory):
    def __init__(self, buffer_size, state_dim, action_shape, obs_space,
                 gamma=0.99, alpha=0.6,max_step=1000,policy=None, eta=None,
                 policy_noise=None, noise_clip=None, max_action=None):
        super(EpisodicMemoryTBP, self).__init__(buffer_size, state_dim, action_shape, obs_space,
                                                gamma, alpha,max_step,policy,policy_noise,noise_clip,max_action)
        del self._q_values
        self._q_values = -np.inf * np.ones((buffer_size + 1, 2))
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.policy = policy
        self.eta = eta

    def compute_approximate_return_double(self, obses, actions=None):
        obses = torch.FloatTensor(obses).to(device)
        actions = torch.FloatTensor(actions).to(device)
        with torch.no_grad():
            target_v1, target_v2 = self.policy.critic_target(obses)
            target_v3, target_v4 = self.policy.critic_target_2(obses)
            target_v = torch.stack([target_v1, target_v2, target_v3, target_v4]).cpu().detach().numpy()
        return target_v

    def update_memory(self, q_base=0, use_knn=False, beta=-1):
        discount_beta = beta ** np.arange(self.max_step)
        trajs = self.retrieve_trajectories()
        for traj in trajs:
            approximate_qs = self.compute_approximate_return_double(self.replay_buffer[traj], self.action_buffer[traj])
            num_q = len(approximate_qs)
            if num_q >= 4:
                approximate_qs = approximate_qs.reshape((2, num_q//2, -1))
                approximate_qs = np.min(approximate_qs, axis=1) 
            else:
                assert num_q == 2
                approximate_qs = approximate_qs.reshape(2, -1)
            approximate_qs = np.concatenate([np.zeros((2, 1)), approximate_qs], axis=1)
            self.q_values[traj] = 0

            rtn_1 = np.zeros((len(traj), len(traj)))
            rtn_2 = np.zeros((len(traj), len(traj)))

            for i, s in enumerate(traj):
                rtn_1[i, 0], rtn_2[i, 0] = self.reward_buffer[s] + \
                                           self.gamma * (1 - self.truly_done_buffer[s]) * (
                                                   approximate_qs[:, i] - q_base)
            for i, s in enumerate(traj):
                rtn_1[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_1[i - 1, :-1]
                rtn_2[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_2[i - 1, :-1]

            if beta > 0:
                double_rtn = [
                    [np.dot(rtn_2[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                        discount_beta[:min(i + 1, self.max_step)]),
                     np.dot(rtn_1[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                         discount_beta[:min(i + 1, self.max_step)])]
                    for i in range(len(traj))]
            else:
                double_rtn = [
                    [rtn_2[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])],
                     rtn_1[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])]] for i
                    in
                    range(len(traj))]
                
            one_step_q = np.array([rtn_1[:, 0], rtn_2[:, 0]]).transpose()
            self.q_values[traj] = np.maximum(np.array(double_rtn),
                                             one_step_q)

    def softmax(self, x):
        y = x - np.max(x)
        y = np.exp(y)
        f_x = y / np.sum(y)
        return f_x
                                        