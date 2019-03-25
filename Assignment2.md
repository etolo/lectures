# Assignment 2

## P1 (max points 25)

Consider a two-player game that proceeds in successive rounds where in each round one player wins and other loses.  The winner of the game is the player who first wins  <img src="https://latex.codecogs.com/svg.latex?\Large&space;d" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> rounds more than the opponent, where <img src="https://latex.codecogs.com/svg.latex?\Large&space;d{\geq}1" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> is a parameter. 

Suppose we model this game so that we consider one of the players. In each round, this player has two available actions, either to invest high or low effort. 

If the player invests high effort in a round, she wins this round with probability  <img src="https://latex.codecogs.com/svg.latex?\Large&space;p_H" title="a" /> , otherwise, she wins this round with probability <img src="https://latex.codecogs.com/svg.latex?\Large&space;p_L" title="a" /> .

<img src="https://latex.codecogs.com/svg.latex?\Large&space;p_L" title="a" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;p_H" title="a" />  are parameters such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;0<p_L<p_H<1" title="x" /> . 

By investing high effort in a round, the player incurs a cost of value <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_H" title="x" />, and otherwise, a cost of value <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_L" title="x" />, in this round.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;c_L" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_H" title="x" /> are parameters such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;0<c_L<c_H" title="x" />.

The player receives a prize of value <img src="https://latex.codecogs.com/svg.latex?\Large&space;R" title="x" /> if winning the game. 

Assume the following values for parameters:
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;d=3" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;p_H=0.55" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;p_L=0.45" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_H=50" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_L=10" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;R=1000" title="x" />

Questions:

1. **[max points 5]** Compute numerically the value function, by assuming perfect knowledge of the environment, for each deterministic policy and rank these policies with respect to the value at the initial state. What is the optimal deterministic policy? Discuss the results. 
2. **[max points 10]** Estimate the action-value function by using the off-policy Monte Carlo estimation method for a threshold policy <img src="https://latex.codecogs.com/svg.latex?\Large&space;\pi" title="x" />
such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;\pi(s)=high" title="x" /> whenever <img src="https://latex.codecogs.com/svg.latex?\Large&space;s<=s^*" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\pi(s)=low" title="x" />  otherwise, for given threshold value <img src="https://latex.codecogs.com/svg.latex?\Large&space;s^*=1" title="x" /> where the behaviour policy, in each state, selects either action equiprobably. Show action values. Can you conclude from the obtained results whether the target policy is optimal? Discuss your answer.
3. **[max points 10]** Compute the optimal policy by using the off-policy Monte Carlo algorithm for the behavior policy that in each state selects either action equiprobably. Show action values and policy. 

In P1-1, use as a termination criteria <img src="https://latex.codecogs.com/svg.latex?\Large&space;||V_{t+1}-V_t||_{\infty}{\leq}1e-6" title="x" />. 

In P1-2 and P2-3, the number of episodes equal to `500,000`. 


## P2 (max points 25)

Consider the same reinforcement learning problem as in P1 but for the following questions:

1. **[max points 5]** Solve the problem by using Q-learning algorithm. Use <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="x" />-greedy with <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon=0.1" title="x" />. 
2. **[max points 5]** Solve the problem by using SARSA algorithm. Use <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="x" />-greedy with  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon=0.1" title="x" />.
3. **[max points 15]** Solve the problem by using on-line TD(<img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="x" />)-algorithm. Use the values of parameters <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda=0.9" title="x" /> and step size <img src="https://latex.codecogs.com/svg.latex?\Large&space;\eta=0.0001" title="x" />.

In all questions, use the number of episodes equal to `500,000`. For each question report the estimated values and policy. 


## P3 (max points 25)

Consider a variant of the reinforcement learning problem which is identical to that in P1 but where the player does not incur a cost by 
investing effort but can invest high effort if she has a sufficient energy. 

The player starts with a given energy level <img src="https://latex.codecogs.com/svg.latex?\Large&space;B" title="x" /> which is decremented by <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_H" title="x" />whenever in a round the player invests high effort. 
If the energy level would become negative after subtracting the value of <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_H" title="x" />, it is set equal to <img src="https://latex.codecogs.com/svg.latex?\Large&space;0" title="x" />. 
In each round, an amount of energy of value <img src="https://latex.codecogs.com/svg.latex?\Large&space;a" title="x" /> is added to the player, independently with probability <img src="https://latex.codecogs.com/svg.latex?\Large&space;p" title="x" />. 

The player can invest high effort in a round only if her energy level is at least <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_H" title="x" />.

The player receives a prize of value <img src="https://latex.codecogs.com/svg.latex?\Large&space;R" title="x" />if winning the game and this is the only reward received.

Use the following setting of parameters:
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;p_H=0.55" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;p_L=0.45" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_H=1" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;R=1000" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;B=10" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;a=2" title="x" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;p=0.2" title="x" />
* step size <img src="https://latex.codecogs.com/svg.latex?\Large&space;\eta=0.001" title="x" />
* Discount parameter <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma=0.9" title="x" />

1. Solve this problem by using SARSA(<img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="x" />) algorithm, for the value of parameter <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda=0.9" title="x" />. Use `500,000` episodes. 
Show the estimated action values for different states and policy. Discuss the results. 


## P4 (max points 25)

Consider the following two-player game of chance. 
Two players, we refer to as players <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;Y" title="x" />, have initial endowments of <img src="https://latex.codecogs.com/svg.latex?\Large&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;y" title="x" /> tokens, respectively. 
Assume that <img src="https://latex.codecogs.com/svg.latex?\Large&space;x,y>0" title="x" /> . The game proceeds over rounds where in each round a dice is rolled. 
If the outcome of the dice is <img src="https://latex.codecogs.com/svg.latex?\Large&space;1,2" title="x" /> or <img src="https://latex.codecogs.com/svg.latex?\Large&space;3" title="x" />, player <img src="https://latex.codecogs.com/svg.latex?\Large&space;Y" title="x" /> loses one token, and otherwise, 
if the outcome is <img src="https://latex.codecogs.com/svg.latex?\Large&space;4,5" title="x" /> or <img src="https://latex.codecogs.com/svg.latex?\Large&space;6" title="x" />, player <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" title="x" /> loses one token. 
The game ends as soon one of the players runs out of tokens. 
The winner of the game is the player who at the end of the game has at least one token left.

Answer the following questions:

1. **[max points 5]** What is the winning probability of player <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" title="x" /> for <img src="https://latex.codecogs.com/svg.latex?\Large&space;x=1" title="x" /> and each value of <img src="https://latex.codecogs.com/svg.latex?\Large&space;y" title="x" />?

2. **[max points 5]** What is the winning probability of player <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" title="x" /> for <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=1" title="x" /> and each value of <img src="https://latex.codecogs.com/svg.latex?\Large&space;x" title="x" />?

3. **[max points 15]** What is the winning probability of player <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" title="x" /> for each value of <img src="https://latex.codecogs.com/svg.latex?\Large&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;y" title="x" />?

**Note:** You need to show derivations for your solutions.  



