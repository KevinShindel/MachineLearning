### Reinforcement 
- Reinforcement learning allows to create AI agents that learn from the env. by interacting with it: learns by trial and error.
- The env. exposes a state to the agent, with a number of possible actions the agent can perform
- After each action, the agent receives feedback: a reward and next state of the env.

### Concept 
-

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


### ☞ Q learning example 
-

### Deep Q learning 
-

### ☞ Deep Q learning example 
-

### Further aspects 
-

### Variants 
-

### ☞ Double deep Q learning example 
-

### Software 
-

### Challenges 
-
