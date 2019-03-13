import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)


bandit_probs = [0.10, 0.40, 0.10, 0.10]


# environment
def step(action):
    rand = np.random.random()  # [0.0,1.0)
    reward = 1.0 if (rand < bandit_probs[action]) else 0.0
    return reward


def decaying_epsilon(c, d, t):
    val = (c * len(bandit_probs)) / (d ** 2 * t)
    return np.minimum(1.0, val)


def epsilon_greedy_policy(epsilon, actions, q_values):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(actions)
    else:
        return np.random.choice([action_ for action_, value_ in enumerate(q_values) if value_ == np.max(q_values)])


def compute_regret(q_star, arms, time_steps):
    probs = [bandit_probs[arm] for arm in arms]
    return np.cumsum(np.ones(time_steps)) * np.amax(q_star) - np.cumsum(probs)


def cumulative_average_mean(r, n):
    return np.cumsum(r) / np.cumsum(np.ones(n))


def plot_rewards(r1, r2, r3, r4, n):
    plt.plot(cumulative_average_mean(r1, n))
    plt.plot(cumulative_average_mean(r2, n))
    plt.plot(cumulative_average_mean(r3, n))
    plt.plot(cumulative_average_mean(r4, n))
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend(['$\epsilon$=0', '0.01', '0.1', 'decaying eps'])
    plt.show()


def plot_regrets(a1, a2, a3, a4):
    plt.plot(a1)
    plt.plot(a2)
    plt.plot(a3)
    plt.plot(a4)
    plt.xlabel('Steps')
    plt.ylim(0.0, 50.0)
    plt.ylabel('Average regret')
    plt.legend(['$\epsilon$=0', '0.01', '0.1', 'decaying eps'])
    plt.show()


def main():
    n_bandits = len(bandit_probs) # = n_actions
    action_space = np.arange(n_bandits)

    n_trials = 50
    timesteps = 1000
    epsilon = [0.0, 0.01, 0.1]
    const = 0.0

    # For decaying epsilon
    n = np.zeros(n_bandits, dtype=np.int)
    q = np.zeros(n_bandits, dtype=np.float)
    h = np.zeros(n_bandits, dtype=np.float)
    arms = np.zeros(timesteps, dtype=np.int)

    # hyper-parameters for decaying eps
    c = 0.5
    d = 0.3
    rewards_decaying = np.zeros(timesteps, dtype=np.float)
    regret_decaying = np.zeros(timesteps, dtype=np.float)

    for trial in range(n_trials):

        for t in range(timesteps):
            eps = decaying_epsilon(c, d, (t+1))
            action = epsilon_greedy_policy(eps, action_space, h)

            r = step(action)

            # updating action counter and Q
            n[action] += 1
            q[action] = q[action] + 1.0 / (n[action] + 1) * (r - q[action])
            # compute preference score
            h[action] = q[action] + const * np.sqrt(np.log(t + 1) / n[action])

            rewards_decaying[t] += r
            arms[t] = action

        regret = compute_regret(bandit_probs, arms, timesteps)
        regret_decaying += regret

    rewards_decaying /= n_trials
    regret_decaying /= n_trials

    # For different epsilon = [0.0, 0.01, 0.1]
    rewards = np.zeros((len(epsilon), n_trials, timesteps), dtype=np.float)
    regrets = np.zeros((len(epsilon), n_trials, timesteps), dtype=np.float)
    eps_counter = 0

    for eps in epsilon:
        n = np.zeros(n_bandits, dtype=np.int)
        q = np.zeros(n_bandits, dtype=np.float)
        h = np.zeros(n_bandits, dtype=np.float)

        for trial in range(n_trials):

            for t in range(timesteps):
                action = epsilon_greedy_policy(eps, action_space, h)

                r = step(action)

                # updating action counter and Q
                n[action] += 1
                q[action] = q[action] + 1.0 / (n[action] + 1) * (r - q[action])
                # compute preference score
                h[action] = q[action] + const * np.sqrt(np.log(t + 1) / n[action])

                rewards[eps_counter, trial, t] += r
                arms[t] = action

            regret = compute_regret(bandit_probs, arms, timesteps)
            regrets[eps_counter, trial, :] += regret
        eps_counter += 1

    rewards = np.mean(rewards, axis=1)
    regrets = np.mean(regrets, axis=1)
    plot_rewards(rewards[0, :], rewards[1, :], rewards[2, :], rewards_decaying, timesteps)
    plot_regrets(regrets[0, :], regrets[1, :], regrets[2, :], regret_decaying)


if __name__ == '__main__':
    main()