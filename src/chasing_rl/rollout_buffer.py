"""Adapted GAE recurrence from onpolicy/utils/shared_buffer.py.
Copyright (c) 2021 MAPPO contributors. MIT; see third_party/MAPPO_LICENSE.
A single team value is stored; both success and finite-horizon timeout end the task.
"""
import numpy as np

def gae(rewards,values,done,gamma,lam):
    if values.shape != (rewards.shape[0]+1,*rewards.shape[1:]) or done.shape != rewards.shape:
        raise ValueError('inconsistent rollout shapes')
    advantages=np.zeros_like(rewards,dtype=np.float32)
    carry=np.zeros(rewards.shape[1:],dtype=np.float32)
    for step in reversed(range(len(rewards))):
        mask=1.-done[step].astype(np.float32)
        delta=rewards[step]+gamma*values[step+1]*mask-values[step]
        carry=delta+gamma*lam*mask*carry
        advantages[step]=carry
    return advantages,advantages+values[:-1]

class RolloutBuffer:
    def __init__(self,length,num_envs):
        self.length,self.num_envs=length,num_envs
        self.obs=np.zeros((length+1,num_envs,3,17),np.float32)
        self.state=np.zeros((length+1,num_envs,21),np.float32)
        self.actions=np.zeros((length,num_envs,3),np.int64)
        self.old_log_prob=np.zeros((length,num_envs,3),np.float32)
        self.team_reward=np.zeros((length,num_envs),np.float32)
        self.team_value=np.zeros((length+1,num_envs),np.float32)
        self.done=np.zeros((length,num_envs),bool)
        self.next_index=0

    def insert(self,obs,state,actions,log_prob,rewards,values,done):
        t=self.next_index
        if t>=self.length: raise RuntimeError('buffer full')
        self.obs[t],self.state[t]=obs,state
        self.actions[t],self.old_log_prob[t]=actions,log_prob
        self.team_reward[t],self.team_value[t],self.done[t]=rewards,values,done
        self.next_index+=1

    def compute_returns(self,last_obs,last_state,last_value,gamma,lam):
        if self.next_index != self.length: raise RuntimeError('buffer incomplete')
        self.obs[-1],self.state[-1],self.team_value[-1]=last_obs,last_state,last_value
        self.advantage,self.return_target=gae(self.team_reward,self.team_value,self.done,gamma,lam)

    def minibatches(self,num_mini_batch):
        count=self.length*self.num_envs
        if not 1<=num_mini_batch<=count: raise ValueError('invalid mini-batch count')
        # Minibatches keep the three actors belonging to a team transition together.
        ids=np.random.permutation(count)
        for ix in np.array_split(ids,num_mini_batch):
            yield {key:getattr(self,key).reshape(count,*getattr(self,key).shape[2:])[ix]
                   for key in ['actions','old_log_prob','advantage','return_target'] } | {
                       'obs':self.obs[:-1].reshape(count,3,17)[ix],
                       'state':self.state[:-1].reshape(count,21)[ix]}

    def after_update(self):
        self.obs[0]=self.obs[-1]
        self.state[0]=self.state[-1]
        self.team_value[0]=self.team_value[-1]
        self.next_index=0
