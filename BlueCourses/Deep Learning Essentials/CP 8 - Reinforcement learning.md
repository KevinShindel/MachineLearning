### Reinforcement 
- Reinforcement learning allows to create AI agents that learn from the env. by interacting with it: learns by trial and error.
- The env. exposes a state to the agent, with a number of possible actions the agent can perform
- After each action, the agent receives feedback: a reward and next state of the env.

### Reinforcement learning
- Q learning attempts to solve the credit assigment problem
- - Propagates rewards back in time, until decision point reached which was actual cause for obtained reward
- When a Q-table is initialized randomly, then its predictions are initially random as well
- - If we pick an action with highest Q-value the action will be random and the agent performs crude "exploration"
- As a Q-func converges, it returns more consistent Q-values and amount of exploration decreases
- - But this exploration is 'greedy', it settles with the first effective strategy it finds
- A simple and effective fix is -egreedy exploration
- - With probability e-choose a random action, otherwise go with the 'greedy' action with the highest Q-value
- An adaptive approach is also possible 


### Q learning 
- Given one run the agent through an env. we can easily calculate the total reward for that episode
- Given than, the total future reward from time point toward can be expressed as
- Because the enviroment is stochastic it is common to use discontinued future reward instead
- A good strategy for an agent would be to always choose an action that maximizes th future reward.
- Define a function representing the maximum discontinued future reward when we perform action a in state s, and continue optimally from there
- But: how can we esimate the score at the end of game?
- - We know just the current state and action, and not the actions and reward coming after that
- - We can't, Q is just a thoretical concept
- We do know it would be optimal to pick the action with the highes Q value in a certain state
- See notebook: "dle-rl_qlearning.ipynb"

### Deep Q learning
- Playing atari with deep reinforcement learning in 2013
- - One could argue that states never occur, we could possibly represent it as a sparse table containing only visited states
- - Even so, most of the states are very rarely visited and it would take a lifetime of the universe for the Q-table to converge
- - Ideally, we would also like to have a good guess for Q-values for the states we have never seen before
- We could represent our Q-func with a neural network, that takes state and action as input and output corresponding Q-value
- According to the network, which action leads to the highes payoff in a given state?

### Experience replay
- Estimate the future reward in each state using Q-learning and approximate the Q-function using a neural network
- It turns out that approximation of Q-values using non-linear function is not very stable
- - Not easy to converge and takes a long time
- Hence, experience replay is typically applied
- - During gameplay all the experiences are stored in a 'replay memory'
- - When training, random mini-batches from replay memory are used instead of most recent transition
- - This breaks the similarity of subsequent training samples, which otherwise might drive network into a local minimum
- - Help avoid neural network to overly adjust its weights for the most recent state which may affect the action output of other states

### ☞ Deep Q learning example
- See notebook: "dle_rl_dqlearning.ipynb"

### Double DQN
- Deep reinforcement learning with double q-learning
- We show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial over-estimations in some games
- The max operator in standard Q-learning and SQN uses the same values both to select and to evaluate an action. This makes it more likely to select over-estimated values, resulting in over-optimistic value estimates. To prevent this, we can decouple the selection from the evaluation. This is the idea behind Double Q-learning
- Two Q-func are independently learned. one function is then used to determine the maximizing action and second to estimate its value.

### Double deep Q-learning example
- See notebook: "dle_rl_ddqlearning.ipynb"

### Further aspects
- Prioritized replay: Select samples that differentiate the most from our current Q-value predictions
- Skipping frames: don't put every frame in memory
- Novelty rewards: add an extra reward when encountering unknown states

### Variants
- Deep Deterministic Policy Gradient (DDPG) 2015
- Async Advantage Actor-Critic (A3C)
- Continuous DQN (CDQN or NAF)
- Cross-Entropy Method (CEM)
- Dueling network SQN (Dueking DQN)
- Deep SARSA
- AlphaGo Zero
- AlphaStar
- AlphaFold

### Software
- keras-rl: Add-on library for Keras containign RL algorithms https://github.com/keras-rl/keras-rl
- Deep-RL-Keras https://github.com/germain-hug/Deep-RL-Keras
- TensorForce https://github.com/tensorforce/tensorforce
- RLlib https://docs.ray.io/en/latest/rllib/index.html

### Challenges
- Faulty Reward Functions in the Wild https://openai.com/index/faulty-reward-functions/
- learning from Human Preferences https://openai.com/index/learning-from-human-preferences/
- RL can prove to be notoriously hard to train
- Novelty, exploration and reward assigment remains a challenge
- As well as creating a good digital twin environment
