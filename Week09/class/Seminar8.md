# Seminar 8: Temporal-Difference Learning

In this seminar, we will implement and evaluate elementary solution methods for reinforcement learning, 
including methods based on Monte Carlo (MC) estimation and time-differencing (TD) methods. 
Specifically, we will consider examples in which we use the following solution methods:

* A. The Random Walk problem: value function estimation using MC and TD(0) methods
* B. The Windy GridWorld problem: SARSA algorithm for estimation of the action value function and control 
* C. The Cliff Walking problem: Q-learning algorithm and comparison with SARSA

The learning methods considered are based on backups which for dynamic programming, Monte Carlo, and temporal-difference learning can be represented as follows:

<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/dp.png" width="300" height="220"> 
<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/mc.png" width="300" height="220"> 
<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/td.png" width="300" height="220">

## A. The Random Walk problem 

The goal of this exercise is to implement and evaluate the TD(0) method and compare with the MC method for the Random Walk problem.

The Random Walk problem is defined in **Example 6.2** in Sutton & Barto. We introduced it in slide 10 of the lecture notes.

![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/random_walk.png)

### The environment 

We first define the environment. 

Following the style of OpenAI Gym, we write a class that contains a step function and a reset function:
```
# actions  
left = 0  
right = 1   
  
class RandomWalk:  
    def __init__(self, initial_state):  
        self.initial_state = initial_state  
        self.state = self.initial_state  
        self.reward = 0.0  
        self.is_terminal = False  
  
    # write step function that returns obs(next state), reward, is_done  
    def step(self, action):  
        if self.state == 5 and action == right:  
            self.state += 1  
            self.is_terminal = True  
            self.reward = 1.0  
        elif self.state == 1 and action == left:  
            self.state -= 1  
            self.is_terminal = True  
            self.reward = 0.0  
        else:  
            if action == left:  
                self.state -= 1  
                self.is_terminal = False  
                self.reward = 0.0  
            else:  
                self.state += 1  
                self.is_terminal = False  
                self.reward = 0.0  
  
        return self.state, self.reward, self.is_terminal  
  
    def reset(self):  
        self.state = self.initial_state  
        self.reward = 0.0  
        self.is_terminal = False  
    return self.state
```
### Random policy 
A policy that takes a random action (0, 1):
```
def take_random_action():
    return np.random.binomial(1, 0.5)
```
### The learning algorithm

Our goal is to estimate the value function using the TD(0) algorithm, which was described in the lecture notes:

<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/td_algo.png" width="800" height="450">

Implement the TD algorithm with step size = 0.1 and no discount:
```
def temporal_difference(value, next_value, r, alpha=0.1):
    return value + alpha * (r + next_value - value)
```
Solve the Random Walk problem, 
```
values = np.zeros(7)  
value_history = [np.copy(values)]  
  
initial_state = 3  
env = RandomWalk(initial_state)
  
episodes = 100  
  
for i in range(1, episodes + 1):  
    state = env.reset()  
    done = False  
  
    while not done:  
        a = random_policy()  
        next_state, r, done = env.step(a)  
        values[state] = temporal_difference(values[state], values[next_state], r)  
        state = next_state  
  
    value_history.append(np.copy(values))  
```

Display the result:
```
true_values = np.zeros(7)  
true_values[1:6] = np.arange(1, 6) / 6.0  
true_values[6] = 1.0  
plot_state_value(value_history, true_values)
```
![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/TD_random_walkzero.png)

To replicate the result in the book, we **assume all rewards are 0**,  the left terminal state has value 0, and the right terminal state has value 1. All other states have initial value of 0.5, i.e., 
```
values = np.zeros(7)    
values[1:6] = 0.5  
values[6] = 1.0
```
![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/TD_random_walk.png)

### Comparison of MC and TD(0)

We now compare the MC and TD(0) methods for value function estimation. To this end, we use the root mean square error (RMS) as the error function.
The RMS is defined for the difference between the estimated value function and the true value function. 
The true value function is the solution of the Bellman equation, and was given in the lectures. 

The following plot compares the average RMS errors over non-terminal states found by TD(0) and constant-<img src="https://latex.codecogs.com/svg.latex?\Large&space;\alpha" title="a" /> MC.  All non-terminal states were initialised to value 0.5: 
<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/rms.png" width="850" height="400">

## B. The Windy GridWorld problem

The goal of this exercise is implement and evaluate SARSA algorithm for the Windy GridWorld problem which is defined as follows.

The Windy GridWorld problem is described in **Example 6.5** in Sutton & Barto. 
![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/windy_grid.png)

The Windy Gridworld problem is a standard gridworld with a crosswind upward through the middle of the grid. The strength of the wind is indicated by the number below each column.  For example, if you are in the cell to the right of the goal, the action left will take you to the cell above the goal.  To be precise, the next cell you will end up in depends on the action taken and the wind in the current cell. 
- An undiscounted episodic task
- Rewards of -1 until the goal is reached

Similarly, define the variables and parameters as defined in the book, 
```
world_height = 7
world_width = 10
wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# actions
up = 0
left = 1
right = 2
down = 3
actions = [up, left, right, down]
```
### The environment
A step function that returns next state, reward and is_done.  **Note** that the next state is the current state moved towards the current action direction + the wind direction of the **current** state.

```
class WindyGridworld:  
    def __init__(self, init_position, goal_position):  
        self.initial_state = init_position  
        self.goal_state = goal_position  
        self.state = self.initial_state  
        self.reward = 0.0  
        self.is_terminal = False  
  
     # return (next state, reward, is_done)  
     def step(self, action):  
         i, j = self.state  
  
         if self.state == self.goal_state:  
             self.reward = 0.0  
             self.is_terminal = True  
         else:  
             if action == up:  
                 self.state = [max(i - 1 - wind_strength[j], 0), j]  
             elif action == left:  
                 # the next state (j-1) is the action + the wind in the previous state (j)  
                 self.state = [max(i - wind_strength[j], 0), max(j - 1, 0)]  
             elif action == right:  
                 self.state = [max(i - wind_strength[j], 0), min(j + 1, world_width - 1)]  
             elif action == down:  
                 self.state = [max(min(i + 1 - wind_strength[j], world_height - 1), 0), j]  
             else:  
                 assert False, "Actions should be in the range of (0, 4)."  
             self.reward = -1.0  
             self.is_terminal = False  
  
          return self.state, self.reward, self.is_terminal  
  
      def reset(self):  
          self.state = self.initial_state  
          self.reward = 0.0  
          self.is_terminal = False  
          return self.state
```
### The policy

Implement an <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="e" />-greedy policy:
```
def epsilon_greedy_policy(q_values, epsilon=0.1):  
    if np.random.binomial(1, epsilon) == 1:  
        return np.random.choice(actions)  
    else:  
        return np.random.choice([action_ for action_, value_ in enumerate(q_values) if value_ == np.max(q_values)])
```

### The algorithm: 
<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/sarsa.png" width="750" height="100">

<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/sarsa_algo.png" width="800" height="450">

```
def sarsa(qsa, next_qsa, r, alpha=0.1, gamma=1.0):  
    return qsa + alpha * (r + gamma * next_qsa - qsa)
```

Solve the Windy GridWorld problem:

```
q_sa = np.zeros((world_height, world_width, len(actions)))  
episodes = 170  
timesteps = []  
  
start_position = [3, 0]  
terminal_position = [3, 7]  
  
env = WindyGridworld(start_position, terminal_position)  
  
for i in range(1, episodes + 1):  
    state = env.reset() 
    is_done = False
    row, col = state  
    # initialise a  
    a = epsilon_greedy_policy(q_sa[row, col, :])  
    timesteps_per_epi = 1  
  
    while not is_done:  
        next_state, r, is_done = env.step(a)  
        row, col = state  
        n_row, n_col = next_state  
        
        next_a = epsilon_greedy_policy(q_sa[n_row, n_col, :])  
        q_sa[row, col, a] = sarsa(q_sa[row, col, a], q_sa[n_row, n_col, next_a], r)  
  
        state = next_state  
        a = next_a  
        timesteps_per_epi += 1  
    timesteps.append(timesteps_per_epi)
```
```
plot_episode_steps(timesteps, episodes)
```
<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/windy_grid_timesteps.png" width="700" height="500">

In this problem, the only way to terminate an episode is to reach the goal state.  The increasing slope of the graph shows that the steps taken by the agent to reach the goal is decreasing per episode. 

**Note** that the policy is unable to converge to the optimal policy after 170 episodes. Let the number of episodes be 1000. 
By the end of 1000 episodes, we choose a greedy policy
```
optimal_policy = np.argmax(q_sa, axis=2)
```

<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/optimal_actions.png" width="850" height="200">

```
>>>
Optimal policy is:
['D' 'L' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'D']
['R' 'U' 'U' 'U' 'R' 'R' 'R' 'D' 'R' 'D']
['R' 'R' 'R' 'R' 'R' 'R' 'R' 'U' 'R' 'D']
['R' 'R' 'R' 'R' 'R' 'R' 'R' 'G' 'L' 'D']
['R' 'D' 'R' 'R' 'L' 'R' 'U' 'D' 'L' 'L']
['R' 'R' 'L' 'R' 'R' 'U' 'U' 'U' 'L' 'D']
['R' 'R' 'R' 'R' 'U' 'U' 'U' 'U' 'D' 'U']
['0' '0' '0' '1' '1' '1' '2' '2' '1' '0']
```

## C. The Cliff Walking problem

In this exercise, our goal is to implement and evaluate Q-learning algorithm for the Cliff Walking problem, and compare with SARSA.

The Cliff Walking problem is described in **Example 6.6** in Sutton & Barto.
 ![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/cliff_walking.png)
- This is a standard undiscounted and episodic gridworld problem with starting and goal states
- Rewards of -1 for each move
- Stepping into the region marked "The Cliff" incurs a reward of -100 and sends the agent instantly back to the start state

The purpose of this example is to compare SARSA and Q-learning, highlighting the difference between on-policy (sarsa) and off-policy (Q-learning) methods. 

### The environment
```
class CliffWalking:  
    def __init__(self, initial_state, goal_state):  
        self.initial_state = initial_state  
        self.goal_state = goal_state  
        self.state = self.initial_state  
        self.reward = 0.0  
        self.is_terminal = False  
  
     def is_cliff(self):  
         cliff = np.zeros((world_height, world_width), dtype=np.bool)  
         cliff[3, 1: -1] = True  
         return cliff[tuple(self.state)]  
  
     def step(self, action):  
         i, j = self.state  
         
         if action == up:  
             self.state = [max(i - 1, 0), j]  
         elif action == left:  
             self.state = [i, max(j - 1, 0)]  
         elif action == right:  
             self.state = [i, min(j + 1, world_width - 1)]  
         elif action == down:  
             self.state = [min(i + 1, world_height - 1), j]  
         else:  
             assert False, "Actions should be in the range of (0, 4)"  
  
         if is_cliff():  
             self.state = self.initial_state  
             self.reward = -100.0  
             self.is_terminal = False  
         elif self.state == self.goal_state:  
             self.state = self.state  
             self.reward = 0.0  
             self.is_terminal = True  
         else:  
            self.reward = -1.0  
            self.is_terminal = False  
         return self.state, self.reward, self.is_terminal  
  
     def reset(self):  
         self.state = self.initial_state  
         self.reward = 0.0  
         self.is_terminal = False  
         return self.state
```
A helper function to check if the current state is in the region of cliff, 
```
def is_cliff(self):  
    cliff = np.zeros((world_height, world_width), dtype=np.bool)  
    cliff[3, 1: -1] = True  
    return cliff[tuple(self.state)]
```
The first two lines create a matrix as below, 

![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/cliff.png)

### The Q-learning algorithm

<img src="https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/q_learning_algo.png" width="750" height="450">

Note that you need to pass all the state-action pairs of the next state,
```  
def q_learning(qsa, next_qs, r, alpha=0.1,, gamma=1.0):  
    return qsa + alpha * (r + gamma * np.max(next_qs) - qsa)
```

It is impossible to produce a smooth reward curve after a single run. We therefore take 50 runs and take the average of the rewards per episode over 500 episodes in total.  
```
start_position = [3, 0]  
goal = [3, 11]

# Create two instances of the environments for comparison  
env_sarsa = CliffWalking(start_position, goal)  
env_q_learning = CliffWalking(start_position, goal)  
  
for r in range(runs):  
    q_sarsa = np.zeros((world_height, world_width, len(actions)))  
    q_qlearning = np.zeros_like(q_sarsa)  
    for i in range(episodes):  
        state_sarsa = env_sarsa.reset()  
        state_q = env_q_learning.reset()  
        done_sarsa = False  
        done_q = False  
        # choose an action based on current state  
        row, col = state_sarsa  
        a_sarsa = eps_greedy_policy(q_sarsa[row, col, :])  
        g_sarsa = 0.0  
        g_q = 0.0  
  
        while not done_sarsa:  
            next_state_sarsa, r_sarsa, done_sarsa = env_sarsa.step(a_sarsa)  
            
            # choose an action for the next state  
            row, col = state_sarsa  
            n_row, n_col = next_state_sarsa  
            next_a_sarsa = eps_greedy_policy(q_sarsa[n_row, n_col, :])  
            g_sarsa += r_sarsa  
            # sarsa updates  
            q_sarsa[row, col, a_sarsa] = sarsa(q_sarsa[row, col, a_sarsa], q_sarsa[n_row, n_col, next_a_sarsa], r_sarsa)  
  
            state_sarsa = next_state_sarsa  
            a_sarsa = next_a_sarsa  
  
        while not done_q:  
            row_q, col_q = state_q  
            a_q = eps_greedy_policy(q_qlearning[row_q, col_q, :])  
            next_state_q, r_q, done_q = env_q_learning.step(a_q)  
            g_q += r_q  
  
            # Q-learning updates, note the second argument  
            n_row_q, n_col_q = next_state_q  
            q_qlearning[row_q, col_q, a_q] = q_learning(q_qlearning[row_q, col_q, a_q],  
            q_qlearning[n_row_q, n_col_q, :], r_q)  
            state_q = next_state_q  
```

The comparison of the two methods:

![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/cliff_walking_comparison.png)

With limited computational power, it's difficult to replicate the result in the book exactly. The worse performance of Q-learning method is due to the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="e" />-greedy policy which results in the agent's occasionally falling off the cliff when the method takes the optimal rather than safer path. 

The following results confirm that SARSA finds the safer path whilst Q-learning converges to the optimal path. 
```
>>>
>Sarsa optimal policy is:
['R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'D']
['U' 'U' 'R' 'R' 'R' 'U' 'U' 'U' 'R' 'R' 'R' 'D']
['U' 'U' 'U' 'R' 'R' 'U' 'U' 'U' 'U' 'L' 'R' 'D']
['U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'G']

>Q-learning optimal policy is:
['L' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'L' 'D' 'D']
['D' 'R' 'R' 'R' 'R' 'D' 'R' 'R' 'R' 'D' 'D' 'D']
['R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'R' 'D']
['U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'U' 'G']
```
If we choose a decaying <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="e" />, both methods will asymptotically converge to the optimal policy. 

## Homework: the off-line <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e" />-return / TD(<img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e" />) algorithm
Consider a larger version of the Random Walk problem. 
Now we have 19 states instead of 5 and the left terminal state gives a reward of -1. 
The starting state is still the centre state. 

**Task**:
- Implement the off-line <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e" />-return / TD(<img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e"/>) algorithm, where the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e" />-return  is defined as
 <img src="https://latex.codecogs.com/svg.latex?\Large&space;R^{\lambda}_t=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}R^{(n)}_t+\lambda^{T-t-1}R_t" title="e" />
 
and TD(<img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e"/>)  error is 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\Delta{V_t(s_t)}=\alpha[R_t^{\lambda}-V_t(s_t)}]" title="e"/>
 
The updating rule is 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;V_{t+1}(s_t)\leftarrow{V_t(s_t)+\Delta{V_t(s_t)}" title="e"/>
 
- Compare the RMS error averaged over the 19 states, over the first 10 episodes,  and over 100 runs (100 different sequence of walks). 
- Plot the averaged RMS error vs <img src="https://latex.codecogs.com/svg.latex?\Large&space;\alpha" title="e"/> values for the following <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e"/> values:
 
 ![alt text](https://gitlab.com/milanv/AI-and-Deep-Learning/raw/master/Seminars/Seminar8/graphs/td_lam.png)

parameter settings: 
```
lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
alphas = [np.arange(0, 1.1, 0.1),
          np.arange(0, 1.1, 0.1),
          np.arange(0, 1.1, 0.1),
          np.arange(0, 1.1, 0.1),
          np.arange(0, 1.1, 0.1),
          np.arange(0, 0.55, 0.05),
          np.arange(0, 0.22, 0.02),
          np.arange(0, 0.11, 0.01)]
```
- *Check the equivalence property in slide 34 of the lectures for off-line updating case - equivalence of forward and backward views. 


**Hints:**
- Implement the algotithm with a function that takes <img src="https://latex.codecogs.com/svg.latex?\Large&space;\alpha" title="e"/> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="e"/> as arguments and iterate through the parameters given above. 
- The same sets of walks are used for all methods. 
