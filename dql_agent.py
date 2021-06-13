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
DECAY = 0.12
NUM_EPISODES = 100
MAX_TIMESTEPS = 200
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
        keras.layers.Dense(units=400, activation='relu'),
        keras.layers.Dense(units=200, activation='relu'),
        keras.layers.Dense(units=output_shape, activation=keras.layers.LeakyReLU(alpha=ALPHA))
    ])
    model.compile(optimizer='adam', loss=keras.losses.Huber())
    return model

def prepare_observation(obs):
    return obs[:,:,2].reshape(INPUT_SHAPE)

def calculate_reward(score, game_steps, done):
    if not done:
        if score < 0: return 0
        return score / (game_steps+10)
        # return 2*(score+1)**2 + game_steps/50  # exploited by standing still
    else:
        return PENALTY


env = Game()
Qnet = model()
#steps_to_udpate_model = 0
epsilon = MAX_EPS

timesteps = []

for episode in range(NUM_EPISODES):

    print('\n\nEPISODE: {:2d} --> epsilon = {:.2f}'.format(episode,epsilon))
    
    
    obs,score,done,info = env.reset()
    
    obs = prepare_observation(obs)
    qvalues = Qnet(obs).numpy().flatten()
    R = calculate_reward(score, 0, done)

    for t in range(MAX_TIMESTEPS):
        #steps_to_udpate_model += 1
        
        if np.random.rand() <= epsilon:
            action = np.random.choice(5)
        else:
            action = np.argmax(qvalues)

        n_obs, score, done, info = env.step(action, t%10==0)
        n_obs = prepare_observation(n_obs)
        n_qvalues = Qnet(n_obs).numpy().flatten()

        R = calculate_reward(score, t+1, done)

        newQ = (1-LR)*qvalues[action] + LR*(R + GAMMA*np.max(n_qvalues))
        target = np.hstack((qvalues[:action],newQ,qvalues[action+1:])).reshape(1,-1)

        Qnet.fit(obs, target, verbose=0)

        if t%10==0:
            print(' - t: {:<4d}- Rt = {:.2f}'.format(t,R))
            print('   -- qv:', qvalues, '- target:', target[0])

        qvalues = n_qvalues
        obs = n_obs

        if done or keyboard.is_pressed('x'):
            break

    timesteps.append(t)
    epsilon = MIN_EPS + (MAX_EPS-MIN_EPS)*np.exp(-DECAY*episode)
    
    if keyboard.is_pressed('x'):
        break

env.close()
keras.models.save_model(Qnet, 'models/agent_dql.hdf5')

plt.figure(figsize=(12,6.5))
plt.title('Episode durations'); plt.xlabel('Episode number'); plt.ylabel('No timesteps'), plt.grid()
plt.plot(timesteps)
plt.savefig('models/fig')
plt.show()

# np.save('models/timesteps',timesteps)