# Seminar 7:  Dynamic Programming and Monte Carlo methods

## Iterative policy evaluation
### Gridworld problem 
The problem is described in the lecture and in **Example 4.1** in Sutton & Barto. 

![alt text](https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/girdworld.png)

- This is a undiscounted episodic task
- Action set for state s: A(s) = {up, down, left, right}
- Actions that would take the agent off the grid leave the state unchanged
- Reward of value -1 for each transition

**Iterative update rule:** Iterative update for the value function whose limit is the solution of the Bellman equation for the value function under a given policy <img src="https://latex.codecogs.com/svg.latex?\Large&space;\pi" title="a"/> is given by

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\begin{align*}V_{k+1}(s)&=\mathbb{E}_{\pi}[r_{t+1}+\gamma{V_k(s_{t+1})|s_t=s]}\\&=\sum_a\pi(s,a)\sum_{s'}P_{s,s'}^a[R_{s,s'}^a+\gamma{V_k(s')}]\end{align*}" title="a" />

for <img src="https://latex.codecogs.com/svg.latex?\Large&space;s\in{S}" title="a"/>.

As we know, algorithms that fall into the category of Dynamic Programming assume that the agent has a prefect knowledge of the environment, i.e., the transition probabilities and expected rewards are known. In order to compute the state values, we first need to learn the environment. 

### The environment
First, enumerate the actions and define the number of states, actions, rows and columns. 
```
up = 0
right = 1
left = 2
down = 3

n_states = 16
n_actions = 4
max_row = 4
max_col = 4
```
The following code is one of the ways to build the GridWorld environment, 
```
def grid_world():
    p = {}
    grid = np.arange(n_states).reshape([max_row, max_col])
    it = np.nditer(grid, flags=['multi_index'])

    with it:
        while not it.finished:
            s = it.iterindex
            row, col = it.multi_index

            p[s] = {a: [] for a in range(n_actions)}

            is_done = lambda x: x == 0 or x == (n_states - 1)
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                # 4 variables: probability of ending up in the next state after action, next state, reward, done
                p[s][up] = [(1.0, s, reward, True)]
                p[s][right] = [(1.0, s, reward, True)]
                p[s][left] = [(1.0, s, reward, True)]
                p[s][down] = [(1.0, s, reward, True)]

            else:
                s_up = s if row == 0 else s - max_row
                s_right = s if col == (max_col - 1) else s + 1
                s_left = s if col == 0 else s - 1
                s_down = s if row == (max_row - 1) else s + max_row

                p[s][up] = [(1.0, s_up, reward, is_done(s_up))]
                p[s][right] = [(1.0, s_right, reward, is_done(s_right))]
                p[s][left] = [(1.0, s_left, reward, is_done(s_left))]
                p[s][down] = [(1.0, s_down, reward, is_done(s_down))]

            it.iternext()
    return p
```
In the end, the transition matrix is returned after the function is called. The transition matrix is essentially a look-up table:

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/transition_prob.png" width="850" height="400"/>

**Warning:** Make sure you are using **Numpy version-1.15.4** or above to use `np.nditer` module.  In Google Colab, you will run into an error - `AttributeError: __enter__`. That's because Google Colab uses an older version of Numpy.  Upgrade your numpy by 
```
!pip install --upgrade numpy
```
in the header, reset the runtime and run the code again.  

### Iterative policy evaluation
Assume that the initial policy is uniform, we then implement the iterative policy evaluation algorithm using this "look-up table": 

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/algo_iterative_policy.png" width="750" height="550"/>


```
for s in range(n_states):  
    v_s = 0.0  
    for a in range(n_actions):  
        current_entry = transition_probs[s][a]
        p_sa = current_entry[0]  
        next_s = current_entry[1]
        reward = current_entry[2]  
        v_s += pi_a * p_sa * (reward + gam * v_old[next_s])
  v_new[s] = v_s
```
```
>>>
After 1 iteration(s): 
[[ 0. -1. -1. -1.] 
[-1. -1. -1. -1.] 
[-1. -1. -1. -1.] 
[-1. -1. -1. 0.]]

After 3 iteration(s):
 [[ 0.     -2.4375 -2.9375 -3.    ]
 [-2.4375 -2.875  -3.     -2.9375]
 [-2.9375 -3.     -2.875  -2.4375]
 [-3.     -2.9375 -2.4375  0.    ]]
 
After 350 iteration(s):
 [[  0.         -13.99999993 -19.9999999  -21.99999989]
 [-13.99999993 -17.99999991 -19.9999999  -19.9999999 ]
 [-19.9999999  -19.9999999  -17.99999991 -13.99999993]
 [-21.99999989 -19.9999999  -13.99999993   0.        ]]

After 407 iteration(s):
 [[  0. -14. -20. -22.]
 [-14. -18. -20. -20.]
 [-20. -20. -18. -14.]
 [-22. -20. -14.   0.]]
```

## Value Iteration
### Gambler's problem
This problem is described in the lecture and in **Example 4.3** in Sutton & Barto. 
- A gambler makes bets on the outcomes of a sequence of coin flips
- The gambler must decide for each coin flip what portion of his capital to stake
- If outcome of the coin flip = heads then: 
   - the gambler wins as much money as he has staked on this flip
- else: 
   - The gambler loses his stake
- The game ends when the gambler reaches his goal of $100 or loses all the money
- Formulated as an undiscounted, episodic, finite MDP problem
- Pr[outcome of coin flip is heads] =p (known parameter)

**States**: There are 101 states,  <img src="https://latex.codecogs.com/svg.latex?\Large&space;s\in\{0,1,2,...,100\}" title="a" />

**Actions**: The actions are stakes, <img src="https://latex.codecogs.com/svg.latex?\Large&space;a\in\{1,2,...,\min(s,100-s)\}" title="a" /> 

We also assume the probability of the coin coming up heads is 0.4. 

```
goal = 100
states = np.arange(goal + 1)

p_head = 0.4
```

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/policy_iteration.png" width="850" height="550"/>

Value iteration can be seen as a truncated version of policy iteration - when policy evaluation is stopped after just one sweep, it's called value iteration. In the policy evaluation step, state values are computed using iterative policy iteration.  
```
# Policy evaluation  
while True:  
    delta = 0.0  
    for s in states[1: goal]:  
        actions = np.arange(min(s, goal - s) + 1)  
        actions_returns = []  
  
        for a in actions:  
            actions_returns.append(p_head * state_values[s + a] + (1. - p_head) * state_values[s - a])  
  
        new_value = np.max(actions_returns)  
        delta = np.maximum(delta, np.abs(state_values[s] - new_value))  
        state_values[s] = new_value  
    if delta < theta:  
        break
```
Once the state values have converged,  the algorithm then chooses a greedy policy that takes the action offering maximum return in the policy improvement step.  
```
# Policy improvement  
for s in states[1: goal]:  
    actions = np.arange(min(s, goal - s) + 1) 
    actions_returns = []  
    for a in actions:  
        actions_returns.append(p_head * state_values[s + a] + (1. - p_head) * state_values[s - a])  
    # round to resemble the figure in the book
    policy[s] = actions[np.argmax(np.round(actions_returns[1: goal], 5))]

```
The upper plot shows the state values computed by iterative policy evaluation and the lower plot shows the final optimal policy. 

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/policy.png" height="750"/> 


## Monte Carlo methods
Blackjack is a card game. The goal is to obtain cards the sum of whose numerical values is as large as possible without exceeding 21. The face cards count as 10 and the ace can count either as 1 or as 11. Consider the version in which each player competes independently against the dealer.  The game begins with 2 cards dealt to both dealer and player. One of the dealer's card is shown and the other is hidden. If the player has 21 immediately, it is called a natural. She then wins unless the dealer also has a natural. In that case, the game is a draw. If the player doesn't start with a natural, she can choose to either 'hit' or 'stick'. If the cards in hand exceeds 21, she loses. If she hits, she receives another card. If she sticks, it becomes the dealer's turn. 

This environment has been built in Gym, which is defined as following:
- Each game is an episode
- The cards are drawn from <img src="https://latex.codecogs.com/svg.latex?\Large&space;deck\in\{1,2,3,4,5,6,7,8,9,10,10,10,10\}" title="a"/>  with replacement
- The two actions are <img src="https://latex.codecogs.com/svg.latex?\Large&space;a\in\{hit=1,stick=0\}" title="a"/> 
- The rewards for win, draw and lose are <img src="https://latex.codecogs.com/svg.latex?\Large&space;r\in\{+1,0,-1\}" title="a"/>
- The observation space has 3 variables: the players current sum,
the dealer's one showing card (1-10 where 1 is ace), and whether or not the player holds a usable ace (0 or 1).

### Monte Carlo prediction
Incremental mean: 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mu_k=\frac{1}{k}\sum_{j=1}^kx_j=\mu_{k-1}+\frac{1}{k}(x_k-\mu_{k-1})" title="a"/>

Incremental MC updates: 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;v^j(s_t)=v^{j-1}(s_t)+\frac{1}{N(s_t)}(G^j_t-v^{j-1}(s_t))" title="a"/>

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;G^j_t" title="a"/>  is the total return at j'th episode and <img src="https://latex.codecogs.com/svg.latex?\Large&space;N(s_t)" title="a"/>  is the number of times state <img src="https://latex.codecogs.com/svg.latex?\Large&space;s_t" title="a"/> was visited - **the every-visit MC method**.

We first consider a simple policy which chooses to hit when the sum of the player's cards in hand is equal to or less than 19 otherwise stick and evaluate it using the every-visit MC method. 
```
def policy(hand_sum):
    if hand_sum > 20:
        return 0
    else:
        return 1

def mc_policy_evaluation(state_count, g, value):  
    return value + (g - value) / state_count
```
The player makes decisions on the basis of the three variables in the observation space: current sum (12-21), the dealer's one showing card (1-10) and whether or not she has a usable ace. This makes for a total of 200 states. We therefore create two matrices to store the states for the two situations - with ace and without ace.  The row in each matrix indicates the player's cards and the column indicates the dealer's card.  

Import the environment and create arrays to store the state values and the number of appearances of each states, 
```
env = gym.make('Blackjack-v0')  
values_usable_ace = np.zeros((10, 10))  
values_no_usable_ace = np.zeros_like(values_usable_ace)  
state_count_ace = np.zeros_like(values_usable_ace)  
state_count_no_ace = np.zeros_like(state_count_ace)  
```

Play the game for a number of episodes, count the state visits and record the observations and rewards:
```
for e in range(episodes):  
    done = False  
    g = []  
    state_history = []  
    obs = env.reset()    

    if obs[0] == 11:  
        done = True  
    else: 
        state_history.append(obs)
    
    while not done:  
        a = policy(obs[0])  
        obs, r, done, info = env.step(a)  
        g.append(r)  
        if done:  
            break  
        state_history.append(obs)  
  
    final_reward = sum(g)  
  
    for player_idx, dealer_idx, ace in state_history:  
        player_idx -= 12  
        dealer_idx -= 1  
  
        if ace:  
           state_count_ace[player_idx, dealer_idx] += 1.0  
           values_usable_ace[player_idx, dealer_idx] = mc_policy_evaluation(state_count_ace[player_idx, dealer_idx], final_reward, values_usable_ace[player_idx, dealer_idx])  
        else:  
           state_count_no_ace[player_idx, dealer_idx] += 1.0  
           values_no_usable_ace[player_idx, dealer_idx] = mc_policy_evaluation(state_count_no_ace[player_idx, dealer_idx], final_reward, values_no_usable_ace[player_idx, dealer_idx])
```

After 10,000 iterations,

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/10000with_ace.png" width="550"/> <img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/10000without_ace.png" width="550"/> 

After 500,000 iterations,

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/500000with_ace.png" width="550"/> <img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/500000without_ace.png" width="550"/> 

### Monte Carlo control

In this section, we will solve the Blackjack problem using MC control.  
 
Firstly, import the environment and create Numpy arrays to store the value and the number of appearance for every **state-action** pair,
```
env = gym.make('Blackjack-v0')
n_action = 2
qsa_with_ace = np.zeros([10, 10, n_action])
qsa_without_ace = np.zeros_like(qsa_with_ace)
a_s_counts_ace = np.zeros_like(qsa_with_ace)
a_s_counts_no_ace = np.zeros_like(a_s_counts_ace)
```
Initialise the policy: stick if the sum of player's cards >= 20 (action = 0), else hit
```
policy_with_ace = np.ones([10, 10], dtype=np.int)
policy_with_ace[7:, :] = 0
policy_without_ace = np.ones([10, 10], dtype=np.int)
policy_without_ace[7:, :] = 0
```
Run a certain number of episodes, evaluate and update the current policy after each episode, 
```
for e in range(episodes):  
    done = False  
    obs = env.reset()  
    state_action_history = []  
    g = []  
    # random first action  
    a = env.action_space.sample()  
  
    if obs[0] == 11:  
        done = True  
    else:  
        state_action_history.append([obs[0], obs[1], obs[2], a])  
  
    while not done:  
        obs, r, done, info = env.step(a)  
        g.append(r)  
        if done:  
            break  
        current_player_idx = obs[0] - 12  
        current_dealer_idx = obs[1] - 1  
        if obs[2]:  
            a = policy_with_ace[current_player_idx, current_dealer_idx]  
        else:  
            a = policy_without_ace[current_player_idx, current_dealer_idx]  
  
        state_action_history.append([obs[0], obs[1], obs[2], a])  
  
    final_reward = sum(g)  
    for player_idx, dealer_idx, ace, action in state_action_history:  
        player_idx -= 12  
        dealer_idx -= 1  
  
        if ace:  
            a_s_counts_ace[player_idx, dealer_idx, action] += 1.0  
            qsa_with_ace[player_idx, dealer_idx, action] = policy_evaluation(a_s_counts_ace[player_idx, dealer_idx, action], qsa_with_ace[player_idx, dealer_idx, action], final_reward)  
            # improve policy  
            policy_with_ace[player_idx, dealer_idx] = np.argmax(qsa_with_ace[player_idx, dealer_idx])  
  
        else:  
            a_s_counts_no_ace[player_idx, dealer_idx, action] += 1.0  
            qsa_without_ace[player_idx, dealer_idx, action] = policy_evaluation(a_s_counts_no_ace[player_idx, dealer_idx, action],  
            qsa_without_ace[player_idx, dealer_idx, action], final_reward)  
            policy_without_ace[player_idx, dealer_idx] = np.argmax(qsa_without_ace[player_idx, dealer_idx])
```
After 10,000 iterations,

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/10000policy_with_ace.png" width="550"/><img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/10000policy_without_ace.png" width="550"/> 

After 500,000 iterations,

<img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/500000policy_with_ace.png" width="550"/> <img src="https://github.com/lse-st449/lectures/blob/master/Week08/class/graphs/500000policy_without_ace.png" width="550"/> 

## Homework: 

### GridWorld
- Implement iterative policy evaluation as shown in the lecture slide
- Print the state values for each iteration
- Use the hyper-parameter settings below
```
# Initial state values - 0s  
state_values = np.zeros(16)  
# Assume a uniform policy  
pi_a = 0.25  
# No discount
gam = 1.0  
theta = 1e-10
```

### Gambler's problem
- Implement value iteration
- The stopping criteria is `theta = 1e-8`
- Replicate the plots 

### Blackjack 
-  Complete the code for both MC prediction and control 
-  Replicate the results


