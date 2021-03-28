import torch
import numpy as np

from networks import Actor, Critic
from params import GAMMA, LAMBDA, CLIP, KL_BETA, ENTROPY_COEF, BATCHES_PER_UPDATE, BATCH_SIZE, VALUE_COEFF

def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


class PPO():
    def __init__(self, state_dim, action_dim, clip, lr):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr)

        self.clip = clip
        self.old_actor = Actor(state_dim, action_dim)

    def _calc_loss(self, state, action, old_log_prob, expected_values, gae):
        new_log_prob, action_distr = self.actor.compute_proba(state, action)
        state_values = self.critic.get_value(state).squeeze(1)

        critic_loss = ((expected_values - state_values) ** 2).mean()

        unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
        if self.clip:
            clipped_ratio = torch.clamp(unclipped_ratio, 1 - CLIP, 1 + CLIP)
            actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()
            entropy_loss = -action_distr.entropy().mean()
        else:
            with torch.no_grad():
                old_log_prob, old_action_distr = self.old_actor.compute_proba(state, action)
            actor_loss = -(unclipped_ratio * gae).mean() - KL_BETA * torch.distributions.kl.kl_divergence(old_action_distr, action_distr).mean()
            entropy_loss = -action_distr.entropy().mean()


        return VALUE_COEFF * critic_loss + actor_loss + entropy_loss * ENTROPY_COEF


    def update(self, trajectories, history=None, history_optimizer=None):
        trajectories = map(compute_lambda_returns_and_gae, trajectories)
        transitions = sum(trajectories, []) # Turn a list of trajectories into list of transitions

        state, action, old_log_prob, target_value, advantage = zip(*transitions)
        state = torch.stack(state)
        action = np.array(action)
        old_log_prob = np.array(old_log_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        
        
        for _ in range(BATCHES_PER_UPDATE):
            if history_optimizer is not None:
                tofit = torch.stack([sa for sa in history[1]])
                _, hc = history[0](tofit.unsqueeze(1))
                hc = torch.cat(hc).reshape(-1).unsqueeze(0)
                hc = hc.expand(BATCH_SIZE, hc.shape[1])

            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            s = state[idx]
            if history_optimizer is not None:
                s = torch.cat([s, hc], axis=1)
            a = torch.tensor(action[idx], dtype=torch.float)
            op = torch.tensor(old_log_prob[idx], dtype=torch.float) # Log probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx], dtype=torch.float) # Estimated by lambda-returns 
            adv = torch.tensor(advantage[idx], dtype=torch.float) # Estimated by generalized advantage estimation 

            loss = self._calc_loss(s, a, op, v, adv)

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            if history_optimizer is not None: history_optimizer.zero_grad()

            if history_optimizer is not None:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            if history_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(history[0].parameters(), 1.)

            if history_optimizer is not None: history_optimizer.step()
            self.actor_optim.step()
            self.critic_optim.step()

        self.old_actor.load_state_dict(self.actor.state_dict())
            
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, log_prob = self.actor.act(state)
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], log_prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.torch")


