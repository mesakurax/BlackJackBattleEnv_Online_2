import pickle
import random
from collections import defaultdict

from StudentCode.BlackJackBattleEnv import BlackJack, SimplePolicy

# 从文件中保存价值函数
def save_policy(V, filename='policy.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(V, f)

# 从文件中加载价值函数
def load_policy(filename='policy.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"在{filename}中未找到策略。返回一个空的价值函数。")
        return defaultdict(float)

# 使用epsilon衰减函数
def epsilon_decay(initial_epsilon, episode, epsilon_decay_episodes):
    epsilon = initial_epsilon * (1 - episode / epsilon_decay_episodes)
    return max(epsilon, 0.01)

# 定义epsilon-greedy策略
def make_epsilon_greedy_policy(Q_table, epsilon, nA):
    def policy(state):
        random_action_prob = epsilon / nA
        greedy_action_prob = 1 - epsilon
        action = random.choices([0, 1], weights=[random_action_prob]*(nA-1) + [greedy_action_prob])[0]
        return action
    return policy

# 使用给定的策略生成一集episode
def generate_one_episode(env, policy):
    state = env.reset()
    trajectory = []
    while True:
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        if done:
            break
    return trajectory

# 首次访问蒙特卡洛控制
def first_visit_mc(env, num_episodes, initial_epsilon=1.2, final_epsilon=0.01, epsilon_decay_episodes=200000, alpha=0.001, gamma=0.6):
    Q_table = load_policy()  # 加载现有的价值表
    nA = 2  # 动作数量（假设0: 停牌, 1: 要牌）

    for episode in range(num_episodes):
        # 根据epsilon_decay函数计算当前轮次的epsilon值
        epsilon = epsilon_decay(initial_epsilon, episode, epsilon_decay_episodes)
        policy = make_epsilon_greedy_policy(Q_table, epsilon, nA)
        episode_trajectory = generate_one_episode(env, policy)
        visited_states = set()
        rewards = []

        for state, action, reward in episode_trajectory:
            rewards.append(reward)
            if (state, action) not in visited_states:
                visited_states.add((state, action))
                G = sum(rewards[i] * (gamma ** i) for i in range(len(rewards)))

                old_value = Q_table.get((state, action), 0)
                new_value = old_value + alpha * (G - old_value)
                Q_table[(state, action)] = new_value

    save_policy(Q_table)  # 保存更新后的价值表
    return Q_table

# 使用训练好的价值表定义一个贪婪策略
def greedy_policy(Q_table, state):
    state_tuple = tuple(state)
    action_values = {a: Q_table.get((state_tuple, a), 0.0) for a in range(2)}
    return max(action_values, key=action_values.get)

if __name__ == '__main__':
    num_episodes = 500000  # 训练轮数
    env = BlackJack(dealer_policy=SimplePolicy(17))
    Q_table = first_visit_mc(env, num_episodes)