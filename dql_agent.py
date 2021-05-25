# first try to implement DQL Agent

from tensorflow.keras import layers
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Flatten
from Game import Game
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as keras

import keyboard


INPUT_SHAPE = (1,380,800,1)
NUM_ACTIONS = 5
LR = 0.7
GAMMA = 0.7
MAX_EPS = 1
MIN_EPS = 0.01
DECAY = 0.2
NUM_EPISODES = 10
MAX_TIMESTEPS = 1000
PENALTY = -1000

def model(input_shape=INPUT_SHAPE[1:], output_shape=NUM_ACTIONS):
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=400, activation='relu'),
        keras.layers.Dense(units=200, activation='relu'),
        keras.layers.Dense(units=output_shape, activation='linear')
    ])
    model.compile(optimizer='adam', loss=keras.losses.Huber())
    return model

def prepare_observation(obs):
    return obs[:,:,2].reshape(INPUT_SHAPE)

def calculate_reward(reward, done):
    distance = reward[0]
    game_steps = reward[1]
    if not done:
        return 2*(distance+1)**2 + game_steps/50
    else:
        return PENALTY


env = Game()
Qnet = model()
#steps_to_udpate_model = 0
epsilon = MAX_EPS

for episode in range(NUM_EPISODES):

    print('\n\nEPISODE: {:2d} --> epsilon = {:.2f}'.format(episode,epsilon))
    
    env.reset()
    obs,reward,done,info = env.step(0)
    obs = prepare_observation(obs)
    qvalues = Qnet(obs).numpy().flatten()
    R = calculate_reward(reward, done)

    for t in range(MAX_TIMESTEPS):
        #steps_to_udpate_model += 1

        if t%5==0:
            print(' - timestep: {:<4d}- reward = {:.2f}'.format(t,R))
        
        if np.random.rand() <= epsilon:
            action = np.random.choice(5)
        else:
            action = np.argmax(qvalues)

        n_obs, reward, done, info = env.step(action)
        n_obs = prepare_observation(n_obs)
        n_qvalues = Qnet(n_obs).numpy().flatten()

        R = calculate_reward(reward, done)

        newQ = (1-LR)*qvalues[action] + LR*(R + GAMMA*np.max(n_qvalues))
        target = np.hstack((qvalues[:action],newQ,qvalues[action+1:])).reshape(1,-1)

        Qnet.fit(obs, target, verbose=0)

        qvalues = n_qvalues
        obs = n_obs

        if done or keyboard.is_pressed('x'):
            break

    epsilon = MIN_EPS + (MAX_EPS-MIN_EPS)*np.exp(-DECAY*episode)
    
    if keyboard.is_pressed('x'):
        break

env.close()
keras.models.save_model(Qnet, 'models/agent1.hdf5')