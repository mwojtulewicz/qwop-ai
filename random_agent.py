import random
import time
from Game import Game

env = Game('game')

for i in range(5):
    action = random.choice([0,1,2,3,4])
    state, reward, done, dict = env.step(action)
    print(f'{i} -- action: {action}, reward: {reward}, done: {done}, dict: {dict}, state_shape={state.shape}')

env.pause()
print('game paused...')
time.sleep(1)
env.unpause()
print('unpaused')
print('end')