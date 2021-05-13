import random
import time
from Game import Game

env = Game()

for i in range(5):
    action = random.choice([0,1,2,3,4])
    state, reward, done, d = env.step(action)
    print(f'{i} -- action: {action}, reward: {reward}, done: {done}, dict: {d}, state_shape={state.shape}')

env.pause()
print('game paused...')
time.sleep(1)
env.unpause()
print('unpaused')
print('end')