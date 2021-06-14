# first try to implement DQL Agent

from time import time
from pytesseract.pytesseract import prepare
from tensorflow.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Flatten
from Game import Game
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as keras
from scipy.special import softmax

import keyboard


INPUT_SHAPE = (-1,90,90)
NUM_ACTIONS = 5
LR = 0.5
GAMMA = 0.99
MAX_EPS = 1
MIN_EPS = 0.1
DECAY = 0.01
NUM_EPISODES = 500
MAX_TIMESTEPS = 200
PENALTY = -10
ALPHA = 0.2

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

def model(input_shape=INPUT_SHAPE, output_shape=NUM_ACTIONS):
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=350, activation='relu'),
        keras.layers.Dense(units=200, activation='relu'),
        keras.layers.Dense(units=output_shape, activation=keras.layers.LeakyReLU(alpha=ALPHA), kernel_initializer='zeros')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

def prepare_observation(obs):
    return obs[:,:,2].reshape(INPUT_SHAPE)

def calculate_reward(score, game_steps, done):
    if not done:
        # if score < 0: return 0
        return int(game_steps==MAX_TIMESTEPS) * (score ** 2) # return score^2 if finished without dying, else 0
        # **2 - game_steps/10  # / (game_steps+10)
        # return 2*(score+1)**2 + game_steps/50  # exploited by standing still
    elif done:
        distance_score = score
        time_score = -(game_steps/(abs(distance_score)+1e5))
        return distance_score + time_score
        # return -10
        

def discount_reward(reward_log):
    for rew in reversed(reward_log):
        pass


env = Game()
Qnet = model()
epsilon = MAX_EPS

timesteps = []
rewards = []
best_distance = []

for episode in range(NUM_EPISODES):

    print('\n\nEPISODE: {:2d} --> epsilon = {:.2f}'.format(episode,epsilon))
    
    obs, score, done, info = env.reset()
    obs = prepare_observation(obs)

    episode_log = []
    best_d = 0
    prev_obs = None

    for t in range(MAX_TIMESTEPS):
        
        x_diff = obs - prev_obs if prev_obs is not None else obs
        prev_obs = obs

        qvalues = Qnet(x_diff).numpy().flatten()

        qvs = softmax(qvalues)
        action = np.random.choice(5,p=qvs)

        # if np.random.rand() <= epsilon:
        #     # action = np.random.choice(5)
        #     qvs = softmax(qvalues)
        #     action = np.random.choice(5,p=qvs)
        # else:
        #     action = np.argmax(qvalues)

        obs, score, done, info = env.step(action, False)
        obs = prepare_observation(obs)

        if score > best_d:
            best_d = score

        R = calculate_reward(score, t+1, done)
        
        log = (x_diff, action, R)
        episode_log.append(log)

        if t%10==0:
            print(' |- t:',t,'a:',action,'qv:',qvalues,'qvs:',qvs)

        if done or keyboard.is_pressed('x'):
            break
    
    rewards.append(R)
    timesteps.append(t)
    best_distance.append(best_d)
    print(f' -- R: {R}, dur: {t}, dist: {best_d}')
    epsilon = MIN_EPS + (MAX_EPS-MIN_EPS)*np.exp(-DECAY*episode)

    inputs  = []
    outputs = []
    
    n_x = obs - prev_obs
    n_qvalues = Qnet(n_x).numpy().flatten()
    while episode_log:
        x, action, R = episode_log.pop()
        qvalues = Qnet(x).numpy().flatten()
        newQ = (1-LR)*qvalues[action] + LR*(R + GAMMA*np.max(n_qvalues))
        target = np.hstack((qvalues[:action],newQ,qvalues[action+1:])).reshape(1,-1)
        
        inputs.append(obs)
        outputs.append(target)

        n_x = x
        n_qvalues = Qnet(n_x).numpy().flatten() 

    X = np.array(inputs).reshape(-1,90,90)
    Y = np.array(outputs).reshape(-1,1,NUM_ACTIONS)
    #Qnet.train_on_batch(X,Y)
    Qnet.fit(X,Y,shuffle=False,batch_size=1,epochs=1)
    
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
