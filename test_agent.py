import random
import time
from Game import Game
from matplotlib import pyplot as plt

env = Game()
actions = list(range(9))
print(actions)

for i in range(21):
    action = random.choice(actions)
    state, reward, done, d = env.step(action)
    print(f'{i} -- action: {action}, reward: {reward}, done: {done}, dict: {d}, state_shape={state.shape}')
    if i%5==0:
        plt.imshow(state)
        plt.title(f'frame: {i} -- action: {action}, reward: {reward}, done: {done}, dict: {d}')
        plt.show()

# env.pause()
# print('game paused...')
# time.sleep(1)
# env.unpause()
# print('unpaused')

# time.sleep(5)
# state, reward, done, d = env.step(action)
# plt.imshow(state)
# plt.show()
# print(state[46, 195])

env.window.close()
print('end')
