import sys, os
from turtle import forward
sys.path.insert(0, os.path.abspath(".."))
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np
from common import helper as h


# Use CUDA for storing tensors / calculations if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialisation function for neural network layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# This class defines the neural network policy
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

        # Initialise a neural network with two hidden layers (64 neurons per layer)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
            )

        # declare log std deviation as a learnable parameter
        self.actor_logstd =  torch.nn.parameter.Parameter(torch.tensor(np.zeros(action_dim), device=device))


    # Do a forward pass to map state to action
    def forward(self, state):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimension of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)
        
        # create action distribution 
        probs = Normal(action_mean, action_std)

        return probs


# Class for the Policy Gradient algorithm
class PG(object):
    def __init__(self, state_dim, action_dim, lr, gamma):

        # Define the neural network policy
        self.policy = Policy(state_dim, action_dim).to(device)

        # Create an optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Set discount factor value
        self.gamma = gamma

        # Simple buffers for action probabilities and rewards
        self.action_probs = []
        self.rewards = []

    def update(self,):

        # Prepare dataset used to update policy
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(device).squeeze(-1) # shape: [batch_size,]
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1) # shape [batch_size,]
        self.action_probs, self.rewards = [], [] # clean buffers
        
        ########## Your code starts here. ##########
        # Hints:
        #   1. compute discounted rewards (use the discount_rewards function offered in common.helper)
        #   2. compute the policy gradient loss
        #   3. update the parameters (backpropagate gradients, do the optimizer step, empty optimizer gradients afterwards so that gradients don't accumulate over updates)

        # compute discounted rewards
        disc_rewards = h.discount_rewards(rewards, self.gamma).unsqueeze(1)

        # normalize rewards
        (std, mean) = torch.std_mean(disc_rewards)
        norm_rewards = (disc_rewards - mean)/std

        # compute policy gradient loss for batch
        loss = - torch.mean(norm_rewards * action_probs)
        
        # backpropagate gradients
        loss.backward()
        self.optimizer.step()

        # empty optimizer gradients
        self.optimizer.zero_grad()

        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {'logstd': self.policy.actor_logstd.cpu().detach().numpy()}


    def get_action(self, observation, evaluation=False):
        """Return action and logprob of this action."""

        # Add batch dimension if necessary
        if observation.ndim == 1: observation = observation[None]

        # Convert observation to a torch tensor
        x = torch.from_numpy(observation).float().to(device)

        ########## Your code starts here. ###########
        # Hint: 
        #   1. when evaluation=True, return mean, otherwise return samples from the distribution created in self.policy.forward() function.
        #   2. notice the shape of action and act_logprob.

        # get action distribution following policy for give observation
        action_distribution = self.policy.forward(x)

        # select action 
        if evaluation == True:
            # mean if testing
            action = action_distribution.mean
        else:
            # random sample otherwise
            action = action_distribution.sample()
        
        # get log probability of chosen action
        act_logprob = action_distribution.log_prob(action)

        ########## Your code ends here. ##########
        
        if observation.ndim == 1: action = action[0]

        return action, act_logprob

    def record(self, action_prob, reward):
        """ Store agent's and env's outcomes to update the agent."""
        self.action_probs.append(action_prob)
        self.rewards.append(torch.tensor([reward]))

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))