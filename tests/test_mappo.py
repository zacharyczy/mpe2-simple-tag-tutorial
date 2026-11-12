"""Independent numerical, environment-boundary and persistence checks."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from chasing_rl.envs.pursuit_env import PursuitEnv
from chasing_rl.policies.actor_critic import Policy
from chasing_rl.rollout_buffer import gae,RolloutBuffer
from chasing_rl.mappo import MAPPO,clipped_policy_loss
from chasing_rl.evaluation import evaluate
from chasing_rl.checkpoint import save_checkpoint,load_checkpoint
from chasing_rl.runner import DEFAULTS

torch.set_num_threads(1)

class MAPPOTests(unittest.TestCase):
    def env(self,**kwargs):
        e=PursuitEnv(**kwargs);self.addCleanup(e.close);e.reset(42,43);return e

    def place(self,e,capture=False):
        for a,p in zip(e.raw.world.agents,[[-.8,-.8],[-.8,.8],[.8,-.8],[.8,.8]]):
            a.state.p_pos=np.array(p,dtype=float);a.state.p_vel=np.zeros(2)
        for b,p in zip(e.raw.world.landmarks,[[0.,0.],[-.4,0.]]):b.state.p_pos=np.array(p)
        if capture:
            e.raw.world.agents[0].state.p_pos=np.array([.93,.8])
            e.raw.world.agents[0].state.p_vel=np.array([-.4,0.])
            e.raw.world.agents[1].state.p_pos=np.array([.67,.8])
            e.raw.world.agents[1].state.p_vel=np.array([.4,0.])

    def test_coincident_collision_is_finite_antisymmetric(self):
        e=self.env(opponent='stay')
        a,b=e.raw.world.agents[:2]
        a.state.p_pos=np.array([1.425,1.425])
        b.state.p_pos=a.state.p_pos.copy()
        force=e.raw.world.get_collision_force(a,b)
        reverse=e.raw.world.get_collision_force(b,a)
        self.assertTrue(np.isfinite(force).all())
        np.testing.assert_allclose(force[0],-force[1])
        np.testing.assert_allclose(force[0],reverse[1])
        e.step([0,0,0])
        self.assertTrue(np.isfinite(e.frame().state).all())

    def test_nonzero_collision_matches_upstream(self):
        from mpe2._mpe_utils.core import World
        e=self.env()
        a,b=e.raw.world.agents[:2]
        a.state.p_pos=np.array([.1,0.]);b.state.p_pos=np.zeros(2)
        np.testing.assert_array_equal(e.raw.world.get_collision_force(a,b),
                                      World.get_collision_force(e.raw.world,a,b))

    def test_capture_last_step_and_single_team_bonus(self):
        e=self.env(max_steps=1,opponent='stay');self.place(e,True)
        self.assertFalse(e.captured())
        t=e.step([0,0,0])
        self.assertTrue(t.success);self.assertFalse(t.timeout)
        self.assertAlmostEqual(t.reward,.99)
        self.assertTrue(all(t.info['native_truncated'].values()))
        self.assertEqual(t.next_frame.obs[0,-1],0.)
        with self.assertRaises(RuntimeError):e.step([0,0,0])

    def test_timeout_shapes_and_partial_observation(self):
        e=self.env(max_steps=1,opponent='stay');self.place(e)
        before=e.frame()
        e.raw.world.agents[1].state.p_vel=np.array([.2,.3])
        after=e.frame()
        np.testing.assert_array_equal(before.obs[0],after.obs[0])
        self.assertFalse(np.array_equal(before.state,after.state))
        self.assertEqual(after.obs.shape,(3,17));self.assertEqual(after.state.shape,(21,))
        t=e.step([0,0,0]);self.assertTrue(t.timeout);self.assertAlmostEqual(t.reward,-.01)

    def test_reset_rejects_initial_capture(self):
        e=self.env();original=e.backend.reset;calls=[]
        def wrapped(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(kwargs.get('seed'))
            if len(calls)==1:
                e.raw.world.agents[0].state.p_pos=e.raw.world.agents[3].state.p_pos.copy()
            return result
        e.backend.reset=wrapped
        f=e.reset(55,66)
        self.assertFalse(e.captured());self.assertGreaterEqual(f.info['reset_attempts'],2)
        self.assertEqual(calls[0],55);self.assertIsNone(calls[1])

    def test_reproducible_rule_and_invalid_actions(self):
        e=self.env(opponent='random');f=self.env(opponent='random')
        np.testing.assert_array_equal(e.reset(17,18).state,f.reset(17,18).state)
        for _ in range(5):
            a=e.step([1,2,3]);b=f.step([1,2,3])
            np.testing.assert_array_equal(a.next_frame.state,b.next_frame.state)
            if a.done:break
        e.reset(17,18)
        for a in [[0,1],[0.,1.,2.],[True,False,True],[0,1,5]]:
            with self.assertRaises(ValueError):e.step(a)

    def test_gae_handcalculated_episode_boundary(self):
        rewards=np.array([[1.],[2.],[3.]],np.float32)
        values=np.array([[.5],[.6],[100.],[.8]],np.float32)
        adv,ret=gae(rewards,values,np.array([[False],[True],[False]]),.9,1.)
        np.testing.assert_allclose(ret[:,0],[2.8,2.,3.72],atol=1e-5)
        np.testing.assert_allclose(adv[:,0],[2.3,1.4,-96.28],atol=1e-5)
        delta,_=gae(rewards,values,np.array([[False],[True],[False]]),.9,0.)
        np.testing.assert_allclose(delta[:,0],[1.04,1.4,-96.28],atol=1e-5)

    def test_single_step_nonterminal_bootstrap(self):
        a,r=gae(np.array([[1.]],np.float32),np.array([[.5],[.8]],np.float32),np.array([[False]]),.9,.95)
        np.testing.assert_allclose(r,[[1.72]],atol=1e-6)

    def test_clipping_sign_cases(self):
        ratio=torch.tensor([.5,.8,1.,1.2,1.5])
        for advantage,expected in [(2.,[1.,1.6,2.,2.4,2.4]),(-2.,[-1.6,-1.6,-2.,-2.4,-3.])]:
            actual=[]
            for r in ratio:
                loss,_=clipped_policy_loss(r.log(),torch.tensor(0.),torch.tensor(advantage),.2)
                actual.append(-loss.item())
            np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_update_ratio_old_data_and_actor_state_separation(self):
        torch.manual_seed(9);np.random.seed(9)
        p=Policy();trainer=MAPPO(p,DEFAULTS|{'ppo_epochs':2})
        b=RolloutBuffer(8,1)
        for i in range(8):
            obs=np.random.randn(1,3,17).astype(np.float32)
            state=np.random.randn(1,21).astype(np.float32)
            actions,old=p.act(obs)
            with torch.no_grad():v=p.value(state).numpy()
            b.insert(obs,state,actions,old,np.array([i%3-.2]),v,np.array([i==3]))
            now,_=p.evaluate_actions(obs,actions)
            np.testing.assert_allclose(np.exp(now.detach().numpy()-old),1.,atol=1e-6)
        b.compute_returns(obs,state,np.array([.1]),.99,.95)
        old_log=b.old_log_prob.copy();old_target=b.return_target.copy();old_adv=b.advantage.copy()
        actor=[x.detach().clone() for x in p.actor.parameters()]
        critic=[x.detach().clone() for x in p.critic.parameters()]
        metrics=trainer.update(b)
        self.assertTrue(all(np.isfinite(list(metrics.values()))))
        self.assertTrue(any(not torch.equal(a,c) for a,c in zip(actor,p.actor.parameters())))
        self.assertTrue(any(not torch.equal(a,c) for a,c in zip(critic,p.critic.parameters())))
        np.testing.assert_array_equal(old_log,b.old_log_prob)
        np.testing.assert_array_equal(old_target,b.return_target)
        np.testing.assert_array_equal(old_adv,b.advantage)
        self.assertEqual(p.actor[0].in_features,17);self.assertEqual(p.critic[0].in_features,21)
        b.after_update();self.assertEqual(b.next_index,0)

    def test_evaluation_freezes_weights_and_preserves_rng(self):
        p=Policy();weights={k:v.clone() for k,v in p.state_dict().items()}
        np_state=np.random.get_state();torch_state=torch.get_rng_state()
        result=evaluate(p,lambda:PursuitEnv(max_steps=3),[1000000,1000001])
        self.assertEqual(result['episodes'],2)
        np.testing.assert_array_equal(np_state[1],np.random.get_state()[1])
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        for k,v in p.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))

    def test_checkpoint_roundtrip(self):
        p=Policy();obs=np.random.randn(3,17).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';save_checkpoint(path,p,DEFAULTS,128,{'test':True})
            loaded,data=load_checkpoint(path)
            self.assertEqual(data['env_steps'],128)
            np.testing.assert_array_equal(p.act(obs,True)[0],loaded.act(obs,True)[0])
            torch.testing.assert_close(p.distribution(obs).probs,loaded.distribution(obs).probs,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
