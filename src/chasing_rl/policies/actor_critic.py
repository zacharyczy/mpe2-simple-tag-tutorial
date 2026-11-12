"""Adapted from MAPPO R_Actor/R_Critic: shared feed-forward policy, team value.
Copyright (c) 2021 MAPPO contributors. MIT; see third_party/MAPPO_LICENSE.
No RNN, PopArt, layer normalization, agent-specific critic or image encoders.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

def mlp(input_dim, output_dim, hidden_size, output_gain):
    layers=[]
    for a,b in [(input_dim,hidden_size),(hidden_size,hidden_size)]:
        linear=nn.Linear(a,b)
        nn.init.orthogonal_(linear.weight, np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.extend([linear,nn.Tanh()])
    last=nn.Linear(hidden_size,output_dim)
    nn.init.orthogonal_(last.weight,output_gain)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers,last)

class Policy(nn.Module):
    def __init__(self, hidden_size=64, device='cpu'):
        super().__init__()
        self.actor=mlp(17,5,hidden_size,.01)
        self.critic=mlp(21,1,hidden_size,1.)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def tensor(self,x,dtype=torch.float32):
        return torch.as_tensor(x,dtype=dtype,device=self.device)

    def distribution(self,obs):
        return Categorical(logits=self.actor(self.tensor(obs)))

    @torch.no_grad()
    def act(self,obs,deterministic=False):
        dist=self.distribution(obs)
        actions=dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions.cpu().numpy(),dist.log_prob(actions).cpu().numpy()

    def value(self,state):
        return self.critic(self.tensor(state)).squeeze(-1)

    def evaluate_actions(self,obs,actions):
        dist=self.distribution(obs)
        return dist.log_prob(self.tensor(actions,torch.long)),dist.entropy()
