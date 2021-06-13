# first try to implement DQL Agent

from time import time
from tensorflow.keras import layers
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Flatten
from Game import Game
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as keras

import keyboard


INPUT_SHAPE = (-1,90,90,1)
NUM_ACTIONS = 5
LR = 0.5
GAMMA = 0.8
MAX_EPS = 1
MIN_EPS = 0.01
DECAY = 0.05
NUM_EPISODES = 200
MAX_TIMESTEPS = 500
PENALTY = -1000
ALPHA = 0.1

hyperparams = {
    'input_shape': INPUT_SHAPE,
    'num_actions': NUM_ACTIONS,
    'learning_rate': LR,
    'gamma': GAMMA,
    'min_epsilon': MIN_EPS,
    'max_epsilon': MAX_EPS,
    'epsilon_decay': DECAY,
    'num_episodes': NUM_EPISODES,
    'max_timesteps': MAX_TIMESTEPS,
    'loss_penalty': PENALTY,
    'lrelu_alpha': ALPHA
}

def model(input_shape=INPUT_SHAPE[1:], output_shape=NUM_ACTIONS):
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=350, activation='relu'),
        keras.layers.Dense(units=200, activation='relu'),
        keras.layers.Dense(units=output_shape, activation='relu')
    ])
    model.compile(optimizer='adam', loss=keras.losses.Huber())
    return model

def prepare_observation(obs):
    return obs[:,:,2].reshape(INPUT_SHAPE)

def calculate_reward(score, game_steps, done):
    if not done:
        # if score < 0: return 0
        return 0  # score  # **2 - game_steps/10  # / (game_steps+10)
        # return 2*(score+1)**2 + game_steps/50  # exploited by standing still
    elif game_steps==MAX_TIMESTEPS:
        return score**2
    else:
        distance_score = score
        time_score = -(game_steps/(abs(distance_score)+1e5))
        return distance_score + time_score

def discount_reward(reward_log):
    discount = 0


env = Game()
Qnet = model()
#steps_to_udpate_model = 0
epsilon = MAX_EPS

timesteps = []
rewards = []
best_distance = []

for episode in range(NUM_EPISODES):

    print('\n\nEPISODE: {:2d} --> epsilon = {:.2f}'.format(episode,epsilon))

    episode_log = []
    best_d = 0
    
    obs,score,done,info = env.reset()
    obs = prepare_observation(obs)
    qvalues = Qnet(obs).numpy().flatten()
    # R = calculate_reward(score, 0, done)
    # print(qvalues, np.argmax(qvalues))

    for t in range(MAX_TIMESTEPS):
        #steps_to_udpate_model += 1
        
        if np.random.rand() <= epsilon:
            action = np.random.choice(5)
        else:
            action = np.argmax(qvalues)

        n_obs, score, done, info = env.step(action, t%5==0)
        n_obs = prepare_observation(n_obs)
        qvalues = Qnet(n_obs).numpy().flatten()

        if score > best_d:
            best_d = score

        R = calculate_reward(score, t+1, done)
        
        log = (obs, action, R)
        episode_log.append(log)

        # newQ = (1-LR)*qvalues[action] + LR*(R + GAMMA*np.max(n_qvalues))
        # target = np.hstack((qvalues[:action],newQ,qvalues[action+1:])).reshape(1,-1)
        # Qnet.fit(obs, target, verbose=0)

        # qvalues = n_qvalues
        obs = n_obs

        if t%10==0:
            print(' - t: {:<4d}- Rt = {:.2f}'.format(t,R))
            # print('   -- qv:', qvalues, '- target:', target[0])

        if done or keyboard.is_pressed('x'):
            break
    
    rewards.append(R)
    timesteps.append(t)
    best_distance.append(best_d)
    print(f'--- R: {R}, dur: {t}, dist: {best_d}')
    epsilon = MIN_EPS + (MAX_EPS-MIN_EPS)*np.exp(-DECAY*episode)

    x = []
    y = []
    
    n_obs = obs
    n_qvalues = Qnet(n_obs).numpy().flatten()
    while episode_log:
        obs, action, R = episode_log.pop()
        qvalues = Qnet(obs).numpy().flatten()
        newQ = (1-LR)*qvalues[action] + LR*(R + GAMMA*np.max(n_qvalues))
        target = np.hstack((qvalues[:action],newQ,qvalues[action+1:])).reshape(1,-1)
        
        # Qnet.fit(obs, target, verbose=0)
        x.append(obs)
        y.append(target)

        n_obs = obs
        n_qvalues = Qnet(n_obs).numpy().flatten() 

    x = np.array(x).reshape(INPUT_SHAPE)
    y = np.array(y).reshape(-1,1,NUM_ACTIONS)
    Qnet.fit(x,y,epochs=1)
    
    if keyboard.is_pressed('x'):
        break

env.close()
keras.models.save_model(Qnet, 'models/agent_dql.hdf5')

plt.figure(figsize=(12,6.5))
plt.title('Episode durations'); plt.xlabel('Episode number'); plt.ylabel('No timesteps'), plt.grid()
plt.plot(timesteps)
plt.savefig('models/duration')
plt.show()

plt.figure(figsize=(12,6.5))
plt.title('Episode final rewards'); plt.xlabel('Episode number'); plt.ylabel('Final reward'), plt.grid()
plt.plot(rewards)
plt.savefig('models/rewards')
plt.show()

plt.figure(figsize=(12,6.5))
plt.title('Episode distance'); plt.xlabel('Episode number'); plt.ylabel('Distance'), plt.grid()
plt.plot(best_distance)
plt.savefig('models/distance')
plt.show()

# np.save('models/timesteps',timesteps)