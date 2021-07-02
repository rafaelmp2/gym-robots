# gym-robots

Naive collection of simple matrix gym-compatible environments. <p>
**Under construction.** Just general baselines being used for some experiments, there are still TODOs.

## Installation

```bash
git clone https://github.com/rafaelmp2/gym-robots.git
cd gym-robots
pip install -e .
```
  
## Simple example of use
```python
  import gym
  import gym_robots

  ep_reward = 0

  env = gym.make('robots-v2', n_agents=1)
  for ep_i in range(100000):
      done_n = [False for _ in range(env.n_agents)]
      ep_reward = 0

      env.seed(ep_i)
      obs_n = env.reset()

      while not all(done_n):
          action_n = [env.action_space[agent_id].sample() for agent_id in range(env.n_agents)]
          obs_n, reward_n, done_n, info = env.step(action_n)
          ep_reward += sum(reward_n)
          env.render()
      print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
  env.close()
```

