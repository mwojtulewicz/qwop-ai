import time
from tensorflow import keras
import numpy as np
from Game import Game

import keyboard

INPUT_SHAPE = (-1,90,90,2)
NUM_ACTIONS = 5

def prepare_observation(obs):
    return obs[:,:,2].reshape(INPUT_SHAPE[:-1])

fname = 'models/knee_scraping/network_checkpoint_ep33.h5'
agent = keras.models.load_model(fname)

env = Game()

epsilon = 0.1

obs, _, _, _ = env.reset()
obs = prepare_observation(obs)
prev_obs = obs
x_t = np.stack((prev_obs,obs),axis=-1)

while not keyboard.is_pressed('x'):
    qvalues = agent(x_t).numpy().flatten()
    print(qvalues)
    qvalues[0] /= 2
    if np.random.rand() <= epsilon:
        action = np.random.choice(5)
    else:
        action = np.argmax(qvalues)

    obs, _, done, _ = env.step_test(action, update_score=False)
    obs = prepare_observation(obs)

    x_t_1 = np.stack((prev_obs, obs), axis=-1)

    x_t = x_t_1
    prev_obs = obs

    if done:
        obs, _, _, _ = env.reset()
        obs = prepare_observation(obs)
        prev_obs = obs
        x_t = np.stack((prev_obs,obs),axis=-1)


env.close()
