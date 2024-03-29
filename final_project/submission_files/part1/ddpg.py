import sys, os
sys.path.insert(0, os.path.abspath(".."))
import copy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from common import helper as h
from common.buffer import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            #nn.Linear(state_dim, 256), nn.ReLU(),
            #nn.Linear(256, 256), nn.ReLU(),
            #nn.Linear(256, action_dim)
            nn.Linear(state_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            #nn.Linear(state_dim+action_dim, 256), nn.ReLU(),
            #nn.Linear(256, 256), nn.ReLU(),
            #nn.Linear(256, 1))
            nn.Linear(state_dim+action_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]


class DDPG(object):
    def __init__(self, state_shape, action_dim, max_action, actor_lr, critic_lr, gamma, tau, batch_size, buffer_size=1e6):
        state_dim = state_shape[0]
        self.action_dim = action_dim
        self.max_action = max_action
        self.pi = Policy(state_dim, action_dim, max_action).to(device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=actor_lr)

        self.q = Critic(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
    

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transiton

        if self.buffer_ptr > self.random_transition: # update once have enough data
            for _ in range(update_iter):
                info = self._update()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info


    def _update(self,):
        batch = self.buffer.sample(self.batch_size, device=device)

        ########## Your code starts here. ##########
        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using h.soft_update_params() (See the DQN code)
        
        # compute Q target
        a_next = self.pi_target(batch.next_state)
        q_next =  self.q_target(batch.next_state, a_next)
        q_tar = batch.reward + self.gamma * q_next * batch.not_done

        # compute critic loss
        q_policy = self.q(batch.state, batch.action)
        critic_loss = torch.mean(torch.square(q_tar.detach() - q_policy))

        # update critic's parameter
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # compute actor loss
        a_policy_new = self.pi(batch.state)
        q_policy_new = self.q(batch.state, a_policy_new)
        actor_loss = - torch.mean(q_policy_new)

        # update actor's parameters
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # update target networks
        h.soft_update_params(self.q, self.q_target, self.tau)
        h.soft_update_params(self.pi, self.pi_target, self.tau)

        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {}

    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        if self.buffer_ptr < self.random_transition and not evaluation: # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim)
        else:
            expl_noise = 0.1 * self.max_action # the stddev of the expl_noise if not evaluation
            
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action's shape is correct.

            # caculate action to execute
            action = self.pi(x)

            # if not evaluation, add gaussain noise
            if not evaluation:
                action += expl_noise * torch.randn(action.shape, device=device)

            # get correct action shape
            action = action.squeeze(0)

            ########## Your code ends here. ##########

        return action, {} # just return a positional value


    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    
    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        '''
        self.pi.load_state_dict(torch.load(f'{filepath}/actor.pt'))
        self.q.load_state_dict(torch.load(f'{filepath}/critic.pt'))
        '''
        d = torch.load(filepath)
        self.pi.load_state_dict(d['pi'])
        self.q.load_state_dict(d['q'])
        self.pi_target.load_state_dict(d['pi_target'])
        self.q_target.load_state_dict(d['q_target'])
        
        print('Successfully loaded model from {}'.format(filepath))
    
    def save(self, filepath):
        '''
        torch.save(self.pi.state_dict(), f'{filepath}/actor.pt')
        torch.save(self.q.state_dict(), f'{filepath}/critic.pt')
        '''
        torch.save({
            'pi': self.pi.state_dict(), 
            'pi_target': self.pi_target.state_dict(),
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
        }, filepath)

        print('Successfully saved model to {}'.format(filepath))