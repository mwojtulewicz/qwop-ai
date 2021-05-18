import random
import time
from Game import Game
from matplotlib import pyplot as plt

env = Game()

for i in range(15):
    action = random.choice([0,1,2,3,4])
    state, reward, done, d = env.step(action)
    print(f'{i} -- action: {action}, reward: {reward}, done: {done}, dict: {d}, state_shape={state.shape}')

env.pause()
print('game paused...')
time.sleep(1)
env.unpause()
print('unpaused')


time.sleep(5)

state, reward, done, d = env.step(action)

plt.imshow(state)
plt.show()

print(state[46, 195])

env.window.close()
print('end')
