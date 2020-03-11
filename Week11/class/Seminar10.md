# Seminar 10: Policy-gradient methods

In this exercise, we will implement and evaluate policy-gradient methods for several Atari 2600 games. 
We will see how one can apply policy-gradient methods and use different tricks to make the learning method more robust with respect to stability. 
Specifically, we will consider tricks such as experience replay and using a fixed policy parameter estimate for a number of steps. 
We explained these tricks in the lecture. 

## DQN 
One of the biggest chanllenges in RL is the unstable training process in practice.  One cause of instability in RL is the correlations of observations used for estimation of value and policy functions. 
This problem is dealt with *experience replay* which stores sequences of observations (e.g. state, action and reward) into a memory and uses random samples 
(minibatches) from this memory for learning purposes. Random sampling of minibatches makes correlations of used observation sequences smaller.  

### Experience replay 
`class Memory` allocates a finite-size memory buffer to store training data. The methods under this class include `add`(adds new observations to memory),  `initialise`(initialises training data before training the networks),  and `sample` (samples a mini-batch for training). 
```
class Memory:
    def __init__(self, size, obs_dims, batch_size, history_length):
        self.memory_size = size
        self.observation_dims = obs_dims
        self.batch_size = batch_size
        self.history_length = history_length
        self.state_shape = [self.history_length, self.observation_dims[0], self.observation_dims[1]]
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.observations = np.empty([self.memory_size] + self.observation_dims, dtype=np.uint8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.__count = 0
        self.__current = 0

        self.prestates = np.empty([self.batch_size] + self.state_shape, dtype=np.uint8)
        self.nextstates = np.empty([self.batch_size] + self.state_shape, dtype=np.uint8)

    @property
    def get_count(self):
        return self.__count

    @property
    def get_current(self):
        return self.__current

    def add(self, next_state, action, r, terminal):
        processed_state = preprocess(next_state)
        clipped_r = clipping_reward(r)

        self.actions[self.__current] = action
        self.rewards[self.__current] = clipped_r
        self.observations[self.__current, ...] = processed_state
        self.terminals[self.__current] = terminal
        # maximum is memory size
        self.__count = max(self.__count, self.__current + 1)
        # a finite memory size
        self.__current = (self.__current + 1) % self.memory_size

    def initialise(self, size, env):
        while True:
            env.reset()
            is_done = False
            initial_lives = 5
            while not is_done:
                action = env.action_space.sample()
                next_state, r, is_done, info = env.step(action)

                current_lives = info['ale.lives']

                if current_lives < initial_lives:
                    is_done = True
                    r = 0.0

                self.add(next_state, action, r, is_done)

            if self.__count == size:
                break

    def get_initial_state(self, state):
        processed_state = preprocess(state)
        states = [processed_state for _ in range(self.history_length)]
        return tf.stack(states)

    def get_states_in_training(self, idx, frames):
        next_states = frames[idx - self.history_length:idx]
        return tf.stack(next_states)

    # stack 4 frames
    def get_state(self, index):
        assert self.__count > 0, "Memory empty."
        assert index >= self.history_length - 1, "Index must be greater than or equal to 3."
        return self.observations[(index - (self.history_length - 1)):(index + 1), ...]

    def sample(self):
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = np.random.randint(self.history_length, self.__count - 1)
                if index < self.history_length:
                    continue
                if self.__current <= index <= self.__current + self.history_length:
                    continue
                if self.terminals[index - self.history_length:index].any():
                    continue
                break

            self.prestates[len(indexes), ...] = self.get_state(index - 1)
            self.nextstates[len(indexes), ...] = self.get_state(index)
            indexes.append(index)

        action = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, action, rewards, terminals, self.nextstates
```

### Q-network

Another source that may cause instability is the correlation between the estimates of the action-value function and the target. 
For example, the DQN loss function at iteration <img src="https://latex.codecogs.com/svg.latex?\Large&space;t" title="q-learning-target" /> is defined as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_t(\theta_t)=\mathbb{E}_{(s,a,r,s'){\sim}U(D)}[(r+{\gamma}\max_{a'}\hat{Q}(s',a';\theta_t)-\hat{Q}(s,a;\theta_t))^2]" title="q-learning-target" />

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;r+{\gamma}\max_{a'}\hat{Q}(s',a';\theta_t)" title="q-learning-target" /> is the Q-learning target. 
In DQN, both <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s',a';\theta_t)" title="target" /> that estimates the Q-learning target and 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s,a;\theta_t)" title="target" /> that generates actions are parametrised using convolutional neural 
networks where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta" title="target" /> are the network parameters. 
The network that estimates the Q-learning target is usually called *target network* and the one that generates actions is called *policy network*. 

The correlations between the estimates of action-value function and the target refer to the fact that the estimates computed by the two networks share the same set of parameters.  DQN gets around this problem by freezing the target network, i.e., 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_t(\theta_t)=\mathbb{E}_{(s,a,r,s'){\sim}U(D)}[(r+{\gamma}\max_{a'}\hat{Q}(s',a';\theta_t^{-})-\hat{Q}(s,a;\theta_t))^2]" title="q-learning-target" />

During training, the policy network parameters <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta_t" title="q-learning-target" /> 
are updated per time step, while the target network parameters <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta_t^{-}" title="q-learning-target" /> are only updated every <img src="https://latex.codecogs.com/svg.latex?\Large&space;N" title="q-learning-target" /> steps. 

Implementation of the Q-network:
```
class QNetwork(tf.keras.Model):
    def __init__(self, n_actions, gamma):
        super(QNetwork, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma

        self.normaliser = tf.keras.layers.Lambda(lambda x: x / 255.0)
        self.conv_layer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), activation='relu')
        self.conv_layer2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), activation='relu')
        self.conv_layer3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),activation='relu')
        
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_dense_layer = tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2), activation='relu')
        self.output_layer = tf.keras.layers.Dense(n_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 84, 84, 4])
        normalised = self.normaliser(inputs)

        conv1 = self.conv_layer1(normalised)
        conv2 = self.conv_layer2(conv1)
        conv3 = self.conv_layer3(conv2)

        flat = self.flatten(conv3)
        h1 = self.hidden_dense_layer(flat)
        outputs = self.output_layer(h1)
        return outputs
```

### Loss function

Huber loss:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e384efc4ae2632cb0bd714462b7c38c272098cf5" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.171ex; width:35.901ex; height:7.509ex;" alt="L_{\delta }(a)={\begin{cases}{\frac  {1}{2}}{a^{2}}&amp;{\text{for }}|a|\leq \delta ,\\\delta (|a|-{\frac  {1}{2}}\delta ),&amp;{\text{otherwise.}}\end{cases}}">

```
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    condition  = tf.keras.backend.abs(error) < clip_delta
 
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(condition, squared_loss, linear_loss)
```

### The algorithm

So far we have completed the three major pieces of the algorithm provided in the paper:

![](https://github.com/lse-st449/lectures/raw/master/Week11/class/graphs/algorithm.png)

**Note** that we do **not** optimise w.r.t the parameters of the target network. Instead, we copy the parameters of the policy network to the target network after a certain number of updates of the policy network.

Here's the initialisation of the two networks:
```
policy_network = QNetwork(n_actions, discount)
target_network = QNetwork(n_actions, discount)
target_network.set_weights(policy_network.get_weights())
```
Make sure that the two networks have the same initialisation.  

### Training process

#### Hyperparameter settings:

![](https://github.com/lse-st449/lectures/raw/master/Week11/class/graphs/hyperparam.png)
#### Results
Training curves tracking the agent’s average score and average predicted action-value.

![](https://github.com/lse-st449/lectures/raw/master/Week11/class/graphs/dqn_results.png)

**Figure a**: each point is the average score achieved per episode on Space Invaders. **Figure b**: average score achieved per episode for Seaquest. 

**Figure c**: average predicted action-value on a held-out set of states on Space Invaders. **Figure d**:  average predicted action-value on Seaquest.

Note that each epoch contains 50000 minibatch weight updates which is roughly 30 minutes of training time-it is not clear how many GPUs the model is trained on.  

Implementation of the algorithm:
```
for e in range(episodes):
    score = 0.0
    ep_counts = 0
    update_counts = 0
    print('Episode:', e + 1)
    print('Count:', experience.get_count)
    
    while update_counts < 50000:
      state = env.reset()
      is_done = False
      stacked_frames = []
      processed_state = preprocess(state)
      stacked_frames.append(processed_state)
      
      while not is_done:
        if len(stacked_frames) >= history_len:
            # get idx
            current_idx = len(stacked_frames)
            next_states = experience.get_states_in_training(current_idx, stacked_frames)
        else:
            next_states = experience.get_initial_state(state)

        q_values = policy_network.call(next_states, training=False)
        q_values = np.squeeze(q_values)
        epsilon = compute_epsilon(starting_epsilon, training_iterations)
        action = epsilon_greedy_policy(epsilon, actions, q_values)
        
        next_s, r, is_done, _ = env.step(action)
        processed_next_s = preprocess(next_s)
        stacked_frames.append(processed_next_s)

        experience.add(next_s, action, r, is_done)
        score += r
        update_counts += 1
        training_iterations += 1
         
        if is_done:
          ep_counts += 1
        
        pre_states, action_values, rewards, terminals, post_states = experience.sample()
      
        with tf.GradientTape() as tape:
          q = policy_network.call(pre_states, training=True)
          one_hot_action = tf.one_hot(action_values, n_actions, 1.0, 0.0)
          current_q = tf.reduce_sum(tf.multiply(q, one_hot_action), axis=1)
            
          target = target_network.compute_target(post_states, rewards, terminals)
          loss = huber_loss(target, current_q)
          
        grads = tape.gradient(loss, policy_network.variables)
        optimiser.apply_gradients(zip(grads, policy_network.variables))
        loss_list.append(loss)
        
        if len(loss_list) % 10000 == 0 and len(loss_list) > 0:
          print("Update target network")
          target_network.set_weights(policy_network.get_weights()) 
```

## Policy-gradient methods

### Pong game 

<img src="https://github.com/lse-st449/lectures/raw/master/Week11/class/graphs/pong_game.png" width="750" height="500"/>

- The agent controls one of the paddles to play against an AI player
- Actions are move up or down
- A +1 reward if the ball goes pass the opponent and a -1 reward if the agent misses the ball
- The game terminates if either player reaches a score of 21. 

This part of the seminar is inspired by Andrej Kapathy's blog post [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/). 

### Policy objective function
- Instead of approximating action-value function <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{Q}(s,a;w)=f(\phi(s,a),w)" title="q-learning-target" />, we can also approximate the policy <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{\pi}(s,a;\theta)=g(\phi(s,a),\theta)" title="q-learning-target" />
- Goal: given <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{\pi}(s,a;\theta)" title="q-learning-target" />, find the best <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta" title="q-learning-target" />
- Problem: How to evaluate the quality of policy <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{\pi}(s,a;\theta)" title="q-learning-target" />?
- One solution:  using the average reward per time step under a given policy:
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;J_{avR}(\theta)=\mathbb{E}_{\pi_\theta}[r]=\sum_s\mu^{\pi_\theta}(s)\sum_a{\pi_\theta}(s,a)R_s^a" title="q-learning-target" />

- The gradient of the objective function is 
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;{\nabla}_{\theta}J_{avR}(\theta)=\mathbb{E}_{\pi_\theta}[{\nabla}_{\theta}{\log}\pi_{\theta}(s,a){\cdot}r]" title="q-learning-target" />

### Intuition

#### Negative log-likelihood and cross-entropy
Mathematically, the negative log-likelihood is equivalent to cross-entropy.  
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\begin{align*}\mathrm{NLL}&=-\sum_{i=1}^N{\log}q(x_i;\theta)\\&=-\sum_{x{\in}X}p(x){\log}q(x;\theta)\\&=H(p,q)\end{align*}" title="q-learning-target" />

In supervised learning, there are image data <img src="https://latex.codecogs.com/svg.latex?\Large&space;x_i" title="q-learning-target" /> and corresponding label <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_i" title="q-learning-target" />.  We find the best parameters by minimising each <img src="https://latex.codecogs.com/svg.latex?\Large&space;-{\log}q(y_i|x_i;\theta)" title="q-learning-target" />, which is equivalent to minimising the cross-entropy between <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(y_i)" title="q-learning-target" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;q(\hat{y_i}|x_i;\theta)" title="q-learning-target" /> where <img src="https://latex.codecogs.com/svg.latex?\Large&space;p" title="q-learning-target" /> is the distribution of true labels and <img src="https://latex.codecogs.com/svg.latex?\Large&space;q" title="q-learning-target" /> is the distribution of the predictions. 

Policy-gradient methods are very similar to supervised learning except that 1) we do not have the correct labels and 2) we include rewards in the loss function.   The term <img src="https://latex.codecogs.com/svg.latex?\Large&space;{\log}\pi_{\theta}(s,a)" title="q-learning-target" /> is also a mapping from an input to a certain class.  For instance, in the application of Atari games, the inputs are stacks of frames and corresponding outputs are classes of actions (e.g., UP, DOWN and NOOP).  

#### Policy gradients
One solution to the lack of true labels is to substitute the labels for the actions we actually take as true labels.  For example, assume that *actions*<img src="https://latex.codecogs.com/svg.latex?\Large&space;=\{up,down\}" title="q-learning-target" />,  then action <img src="https://latex.codecogs.com/svg.latex?\Large&space;up" title="q-learning-target" /> is labeled as class <img src="https://latex.codecogs.com/svg.latex?\Large&space;0" title="q-learning-target" /> and action <img src="https://latex.codecogs.com/svg.latex?\Large&space;down" title="q-learning-target" /> is labeled as class <img src="https://latex.codecogs.com/svg.latex?\Large&space;1" title="q-learning-target" />.  We have now turned this problem into a supervised learning problem.  

The question is, how do we know if the sequence of actions we take is good? Since the labels themselves are "fake", how do we make sure that we are learning to become better?  

Luckily there is one more piece of information we can use, i.e., the rewards.  We can simply wait until the episode finishes and see how we did by taking the sequence of actions during the game. If taking the action up ends up losing the game, we will find a gradient that discourages the agent to take action up given the specific state in the future.  
However, as there is always a mixture of good and bad actions in every episode, all actions no matter good or bad will be encouraged or discouraged altogether based on the final outcomes.  This is not ideal.  But the hope is that, over a large number of games, the good actions get more positive than negative updates. 

### The algorithm 

<img src="https://github.com/lse-st449/lectures/raw/master/Week11/class/graphs/policy_grad_algo.png" width="750" height="500"/>

Before we start implementing the algorithm, we also need to preprocess the raw data which is of shape 210x160x3.  
The input data is preprocessed into a long vector of size 6400 (80x80). 

The policy is a one-layer feedforward neural network that returns both the logits and the probability of taking one of the actions: 

```
class Policy(tf.keras.Model):  
    def __init__(self, input_size):  
        super(Policy, self).__init__()  
        self.hidden = 200  
        self.input_size = input_size  
  
        self.dense1 = tf.keras.layers.Dense(self.hidden, activation=tf.nn.relu)  
        self.dense2 = tf.keras.layers.Dense(1, activation=None)  
  
    def call(self, inputs, training=None, mask=None):  
        h1 = self.dense1(inputs)  
        logits = self.dense2(h1)  
        prob = tf.sigmoid(logits)  
        return tf.squeeze(logits), tf.squeeze(prob)
```

The probability returned is used to generate actions and logits are used to compute the cross-entropy. 

A helper function that computes the normalised discounted returns:
```
def discount_rewards(r, gamma):  
    r = np.array(r)  
    discounted_r = np.zeros_like(r)  
    running_add = 0  
 
    for t in reversed(range(0, r.size)):  
        # if the game ended, reset the reward sum  
        if r[t] != 0: 
           running_add = 0 
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add  
     discounted_r -= np.mean(discounted_r)
     discounted_r /= np.std(discounted_r) 
     return discounted_r
```

Implementation of the main algorithm:
```
env = gym.make("Pong-v0")  
# mini_batch and total rewards for some number of episodes
mini_batch = []  
total_rewards = 0.0
policy_net = Policy(n_pixels)

for e in range(1, n_episodes+1):  
    state = env.reset()  
    prev_state = None  
    is_done = False  
    rewards = []  
    labels = []  
    inputs = []  
  
    while not is_done:  
        state = preprocess(state)  
        x = state - prev_state if prev_state is not None else tf.zeros((1, n_pixels))  
        prev_state = state  
  
        _, prob = policy_net.call(x, training=False)  
        action = 3 if np.random.uniform() < prob else 2  
        y = 1.0 if action == 3 else 0.0 # fake labels  
        inputs.append(x)  
        labels.append(y)  
  
        state, r, is_done, _ = env.step(action)  
        rewards.append(r)  
        
        if is_done:  
            total_rewards += np.sum(rewards)  
            discount_epr = discount_rewards(rewards, gamma)  
            mini_batch.append([inputs, labels, discount_epr])  
  
            if e % update_frequency == 0:  
                print('Updates:', updates)  
                for (batch, (x, label, reward)) in enumerate(mini_batch):  
                    with tf.GradientTape() as tape:  
                        logits, _ = policy_net.call(x, training=True)  
                        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)  
                        loss = tf.reduce_mean(tf.multiply(reward, cross_entropy))  
  
                    grads = tape.gradient(loss, policy_net.variables)  
                    optimiser.apply_gradients(zip(grads, policy_net.variables))  
                    loss_list.append(loss)  
                updates += 1  
                print('Average reward over 10 episodes:', total_rewards / update_frequency)  
                # reset mini_batch and total rewards for batch_size number of episodes  
                del mini_batch[:]  
                total_rewards = 0.0
```

The optimiser used is 
```
tf.train.RMSPropOptimizer(learning_rate, decay=decay_rate)
```

and hyperparameter settings are 
```
update_frequency = 10
learning_rate = 1e-3  
gamma = 0.99  
decay_rate = 0.99  
n_episodes = 10000  
```

You could also try to construct a convolutional neural network as the policy network. 

