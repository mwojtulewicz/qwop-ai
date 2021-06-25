# Double Deep Q-Learning Agent with Replay Buffer

import time
import json
from Game import Game
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as keras
from scipy.special import softmax

import keyboard


def model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation='relu', input_shape=(90,90,2)),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu'),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=5, activation='linear', kernel_initializer='zeros')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    return model

def prepare_observation(obs):
    return obs[:,:,2].reshape(INPUT_SHAPE[:-1])

def calculate_reward(score_diff, done):
    if done:
        return -10
    elif score_diff > 0:
        return 1 + (score_diff//0.2)
    elif score_diff == 0:
        return -0.1
    elif score_diff < 0:
        return -0.5
        



INPUT_SHAPE = (-1,90,90,2)
NUM_ACTIONS = 5
MIN_BUFFER_LENGTH = 250
MAX_BUFFER_LENGTH = 2500
TARGET_NET_UPDATE_FREQ = 150
TRAIN_FREQ = 5
BATCH_SIZE = 16
LR = 0.7
GAMMA = 0.99
NUM_EPISODES = 200
MAX_TIMESTEPS = 1000
CHECKPOINT_FREQ = 10
MAX_EPS = 1
MIN_EPS = 0.1
DECAY = (MAX_EPS-MIN_EPS)/50

hiperparams = {
    'input_shape': INPUT_SHAPE,
    'num_actions': NUM_ACTIONS,
    'gamma': GAMMA,
    'min_epsilon': MIN_EPS,
    'max_epsilon': MAX_EPS,
    'epsilon_decay_type': 'linear',
    'epsilon_decay_value': DECAY,
    'num_episodes': NUM_EPISODES,
    'max_timesteps': MAX_TIMESTEPS,
    'min_buffer_length': MIN_BUFFER_LENGTH,
    'max_buffer_length': MAX_BUFFER_LENGTH,
    'target_net_update_freq': TARGET_NET_UPDATE_FREQ,
    'train_freq': TRAIN_FREQ,
    'batch_size': BATCH_SIZE,
    'optimizer': {'class':'Adam', 'lr':0.0005, 'loss':'MSE'}
}


env = Game()
Qnet = model()
Qtarget = model()
Qtarget.set_weights(Qnet.get_weights())

replay_buffer = []

epsilon = MAX_EPS

avg_timesteps = []
cum_rewards = []
avg_distances = []
deaths = []

for episode in range(1,NUM_EPISODES+1):

    print('\n\nEPISODE: {:2d} of {:2d} --> epsilon = {:.2f}'.format(episode, NUM_EPISODES, epsilon))
    
    obs, prev_score, _, _ = env.reset()
    obs = prepare_observation(obs)
    prev_obs = obs
    x_t = np.stack((prev_obs,obs),axis=-1)

    r_sum = 0
    n_runs = 1
    dist = 0
    start_time = time.time()


    for t in range(MAX_TIMESTEPS):

        qvalues = Qnet(x_t).numpy().flatten()
        qvs = softmax(qvalues)
        if np.random.rand() <= epsilon:
            action = np.random.choice(5)
            # action = np.random.choice(NUM_ACTIONS,p=qvs)
        else:
            action = np.argmax(qvalues)

        obs, score, done, info = env.step(action, update_score=True)
        obs = prepare_observation(obs)

        x_t_1 = np.stack((prev_obs, obs), axis=-1)
        
        score_diff = score - prev_score
        r_t = calculate_reward(score_diff, done)
        
        trans = (x_t, action, r_t, done, x_t_1)
        replay_buffer.append(trans)

        x_t = x_t_1
        prev_obs = obs
        prev_score = score
        if len(replay_buffer)>MAX_BUFFER_LENGTH:
            replay_buffer.pop(0)
        
        if (len(replay_buffer) > MIN_BUFFER_LENGTH) and (t%TRAIN_FREQ==0):
            print(' |- t:',t,'a:',action,'qv:',qvalues,'qvs:',qvs)
            print(' |  performing SGD', end=' ')
            trans_indicies = np.random.choice(len(replay_buffer), size=BATCH_SIZE)

            X = []
            Y = []

            for i in trans_indicies:
                _xt, _a, _rt, _d, _xt_1 = replay_buffer[i]
                qvalues = Qnet(_xt).numpy().flatten()
                TD = _rt - qvalues[_a]
                if not _d:
                    TD += GAMMA * max(Qtarget(_xt_1).numpy().flatten())
                qvalues[_a] += LR * TD
                target = qvalues.reshape(1,-1)
                # target = np.hstack((qvalues[:action], newQ, qvalues[action+1:])).reshape(1,-1)

                X.append(_xt)
                Y.append(target)
            
            X = np.array(X).reshape(-1,90,90,2)
            Y = np.array(Y).reshape(-1,1,5)
            Qnet.fit(X,Y,batch_size=BATCH_SIZE,epochs=1,verbose=1)

        if t%TARGET_NET_UPDATE_FREQ==0:
            print(' |  updating target network')
            Qtarget.set_weights(Qnet.get_weights())

        r_sum += r_t
        if done:
            obs, prev_score, _, _ = env.reset()
            obs = prepare_observation(obs)
            prev_obs = obs
            x_t = np.stack((prev_obs,obs),axis=-1)
            n_runs += 1
            dist += score

        if t==MAX_TIMESTEPS-1:
            dist += score

        if keyboard.is_pressed('x'):
            break
    
    cum_rewards.append(r_sum)
    avg_timesteps.append(MAX_TIMESTEPS/n_runs)
    avg_distances.append(dist)
    deaths.append(n_runs-1)
    epsilon = max(MIN_EPS, epsilon-DECAY)
    
    print(f' -- cummulated reward: {r_sum}, avg run distance: {dist/n_runs}, deaths: {n_runs-1}, time: {time.time()-start_time}')
    
    if episode%CHECKPOINT_FREQ==0:
        print(f' -- chechpoint {episode} --')
        keras.models.save_model(Qnet, f'models/network_checkpoint_ep{episode}.h5')

    if keyboard.is_pressed('x'):
        break

env.close()
keras.models.save_model(Qnet, f'models/network_checkpoint_ep{episode}.h5')

plt.figure(figsize=(12,6.5))
plt.title('Average run duration'); plt.xlabel('Episode number'); plt.ylabel('No. timesteps'), plt.grid()
plt.plot(avg_timesteps)
plt.savefig('models/duration')
plt.show()

plt.figure(figsize=(12,6.5))
plt.title('Episode cummulated reward'); plt.xlabel('Episode number'); plt.ylabel('Reward'), plt.grid()
plt.plot(cum_rewards)
plt.savefig('models/rewards')
plt.show()

plt.figure(figsize=(12,6.5))
plt.title('Average run distance'); plt.xlabel('Episode number'); plt.ylabel('Distance'), plt.grid()
plt.plot(avg_distances)
plt.savefig('models/distance')
plt.show()

plt.figure(figsize=(12,6.5))
plt.title('Deaths'); plt.xlabel('Episode number'); plt.ylabel('No. deaths'), plt.grid()
plt.plot(deaths)
plt.savefig('models/deaths')
plt.show()

f = open("models/hiperparams.json","w")
json.dump(hiperparams,f,indent=4)
f.close()

f = open("models/model_summary.txt", "w")
Qnet.summary(print_fn=lambda x: f.write(x+'\n'))
f.close()

f = open("models/model_detailed.json","w")
f.write(Qnet.to_json())
f.close()
