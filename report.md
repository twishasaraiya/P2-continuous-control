## Environment

Note that your project submission need only solve one of the two versions of the environment.

Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## DDPG Algorithm

![DDPG Algorithm](https://github.com/twishasaraiya/P2-continuous-control/blob/master/assets/ddpg.png)

As in DQN, replay buffer is used here as well. The actor and critic are updated by sampling a minibatch uniformly from the buffer. Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing the algorithm to benefit from learning across a set of uncorrelated transitions. Here replay buffer of size `1e6` is used


Based on Attempt 4 of benchmark implementation

> instead of updating the actor and critic networks 20 times at every timestep, we amended the code to update the networks 10 times after every 20 timesteps.


```python
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM   = 10        # number of learning passes
```

## Approach

Initially I started with version 1 environment, but the agent learning was very slow and the max score never went above 1. Even after playing with hyperparameters I couldnt get the agent to learn faster. I wasted a lot of gpu hours because of that. 

So I decided to switch to version and started anew. The agent training much more better than version 1. Based on the tips provided by  Alessandro Restagno(DRLND mentor) in our slack community, I made two modifications

1. Add gradient clipping to prevent exploding gradient
```python
    torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
```

2. Add batch normalization

Add 1 layer of batch normalization 
```python
x = F.relu(self.bn1(self.fc1(state)))
```
### Network Architecture

state_size = 33 
action_size = 4

**Actor**

1. Linear(33,400)
2. Batch Normalization
3. relu
4. Linear(400,300)
5. relu
6. Linear(300,4)
7. tanh


**Critic**

1. Linear(33,400)
2. Batch Normalization
3. relu
4. Linear(400+4,300)
5. relu
6. Linear(300,1)

### Hyperparameters

The baseline hyperparameters are taken from [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING](https://arxiv.org/pdf/1509.02971.pdf)

After tweaking the parameters, below are the parameters for which I got best result

![](https://github.com/twishasaraiya/P2-continuous-control/blob/master/assets/hyperparameters.png)

### Result

![](https://github.com/twishasaraiya/P2-continuous-control/blob/master/assets/rewards_plot.png)
