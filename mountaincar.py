from cmath import e
from time import sleep
from urllib import request
import gym
import numpy as np

env = gym.make('MountainCar-v0')
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

DESCRITE_OS_SPACE = [20] * len(env.observation_space.high)
DESCRITE_OS_WIN_SIZE = (env.observation_space.high -
                        env.observation_space.low) / DESCRITE_OS_SPACE
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

EPSILON = 0.5
START_EPSILON_DEKAYING = 1
END_EPSILON_DEKAYING = EPISODES//2
EPSILON_DEKAY_AMOUNT = EPSILON / (END_EPSILON_DEKAYING - START_EPSILON_DEKAYING)

# print(DESCRITE_OS_WIN_SIZE)

q_table = np.random.uniform(
    low=-2, high=1, size=(DESCRITE_OS_SPACE + [env.action_space.n]))

# print(q_table.shape)


def get_descrete_state(state):
    # print("State", state)
    # print("observation_space", env.observation_space.low)
    # print("observation_space", state - env.observation_space.low)
    # print("observation_space", DESCRITE_OS_WIN_SIZE)

    descrete_state = (state - env.observation_space.low)/DESCRITE_OS_WIN_SIZE

    return tuple(descrete_state.astype(np.int32))


for episode in range(EPISODES):
    if(episode % SHOW_EVERY) == 0:
        render = True
    else:
        render = False

    ds = get_descrete_state(env.reset())
    # print(np.argmax(q_table[ds]))

    done = False
    score = 0
    while not done:
        if render:
            env.render()
            sleep(0.01)

        if np.random.random() > EPSILON:
            action = np.argmax(q_table[ds])
        else:
            action = np.random.randint(0, env.action_space.n)

        observation, reward, done, _ = env.step(action)
        score += reward
        new_ds = get_descrete_state(observation)
        # sleep(0.01)
        # print(observation, reward, done)
        if not done:
            max_future_q = np.max(q_table[new_ds])
            current_q = q_table[ds + (action, )]

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * \
                (reward + DISCOUNT * max_future_q)
            q_table[ds + (action, )] = new_q
        elif observation[0] >= env.goal_position:
            q_table[ds + (action, )] = 0
            print("GOAL", episode)
        ds = new_ds

    if END_EPSILON_DEKAYING >= episode >= START_EPSILON_DEKAYING:
        EPSILON -= EPSILON_DEKAY_AMOUNT
