"""Regression checks: reward -> GAE -> PPO must reinforce the successful action."""
import unittest
import numpy as np
import torch
from torch import nn
from chasing_rl.envs.pursuit_env import PursuitEnv
from chasing_rl.policies.actor_critic import Policy
from chasing_rl.rollout_buffer import RolloutBuffer
from chasing_rl.mappo import MAPPO, clipped_policy_loss
from chasing_rl.runner import DEFAULTS

class SignTests(unittest.TestCase):
    def test_gradient_increases_positive_and_decreases_negative_action(self):
        for advantage in [1.,-1.]:
            logits=nn.Parameter(torch.zeros(5))
            optimizer=torch.optim.SGD([logits],lr=.1)
            action=torch.tensor(2)
            old=torch.distributions.Categorical(logits=logits.detach()).log_prob(action)
            before=torch.softmax(logits.detach(),-1)[2].item()
            loss,_=clipped_policy_loss(torch.distributions.Categorical(logits=logits).log_prob(action),old,torch.tensor(advantage),.2)
            optimizer.zero_grad();loss.backward();optimizer.step()
            after=torch.softmax(logits.detach(),-1)[2].item()
            self.assertGreater((after-before)*advantage,0)

    @staticmethod
    def setup_counterfactual(env):
        env.reset(42,43)
        for a,p in zip(env.raw.world.agents,[[0.,0.],[-.8,.8],[-.8,-.8],[.13,0.]]):
            a.state.p_pos=np.array(p);a.state.p_vel=np.zeros(2)
        for b,p in zip(env.raw.world.landmarks,[[.8,.8],[.8,-.8]]):
            b.state.p_pos=np.array(p)
        return env.frame()

    def test_real_capture_reward_through_gae_and_production_optimizer(self):
        torch.set_num_threads(1);torch.manual_seed(41);np.random.seed(41)
        env=PursuitEnv(max_steps=2,opponent='stay')
        policy=Policy();trainer=MAPPO(policy,DEFAULTS|dict(ppo_epochs=1,entropy_coef=0.))
        buffer=RolloutBuffer(4,1)
        actual_rewards=[]
        try:
            for first_action in [2,1]:
                frame=self.setup_counterfactual(env)
                original_obs=frame.obs.copy()
                self.assertGreater(frame.obs[0,12],0)
                for time,action in enumerate([first_action,0]):
                    actions=np.array([[action,0,0]])
                    logp=policy.evaluate_actions(frame.obs[None],actions)[0].detach().numpy()
                    with torch.no_grad():value=policy.value(frame.state[None]).numpy()
                    result=env.step(actions[0]);actual_rewards.append(result.reward)
                    if time==0:
                        self.assertFalse(result.done)
                        # Explicit Euler: action changes velocity now, displacement next step.
                        self.assertAlmostEqual(result.next_frame.obs[0,2],original_obs[0,2])
                    else:
                        self.assertEqual(result.success,first_action==2)
                        self.assertTrue(result.done)
                    buffer.insert(frame.obs[None],frame.state[None],actions,logp,
                                  np.array([result.reward]),value,np.array([result.done]))
                    frame=result.next_frame
            np.testing.assert_allclose(actual_rewards,[-.01,.99,-.01,-.01])
            buffer.compute_returns(frame.obs[None],frame.state[None],np.array([999.]),.99,1.)
            np.testing.assert_allclose(buffer.return_target[:,0],[.9701,.99,-.0199,-.01],atol=1e-6)
            before=policy.distribution(original_obs).probs.detach().numpy()[0]
            trainer.update(buffer)
            after=policy.distribution(original_obs).probs.detach().numpy()[0]
            self.assertGreater(after[2],before[2])
            self.assertLess(after[1],before[1])
        finally:env.close()

    def test_reward_objective_prefers_success_and_earlier_success(self):
        gamma=.99
        failure=-.01*sum(gamma**t for t in range(100))
        successes=[-.01*sum(gamma**t for t in range(T))+gamma**(T-1) for T in range(1,101)]
        self.assertTrue(all(x>failure for x in successes))
        self.assertTrue(all(a>b for a,b in zip(successes,successes[1:])))

if __name__=='__main__': unittest.main()
