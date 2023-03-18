# Implementation

## Environment
The tennis environment contains 2 players, both players are represented as an Agent. The agents have to collaborate in order to adchieve the highest scores. So they ar not playing to win the game, but they are playing to keep the match as long as possible, and thus they have to collaborate. 

### Action space
Each action of the agent consists of 2 continuous values:
- horizontal movement (ranging from -1 to 1), where -1 is towards the net, and vise versa.
- vertical movement (ranging from -1 to 1), where -1 is down and vise versa.

### State space
The state space consists of 8 variables. These variables correspond to the velocity and position of the ball as well of the rackets. The unity environment stacks 3 states, resulting in (3 * 8) 24 variables.

### Goal
Each agent earns a score per episode. The goal of this assignment is to adchieve an average score (of both agents) of 0.5 over 100 consecutive episodes.

# Training
...

## Deep Deterministic Policy Gradient
...

### Hyperparameters
...

## Future Work
...
