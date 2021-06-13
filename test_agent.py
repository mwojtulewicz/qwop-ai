import random
import time
from Game import Game
from matplotlib import pyplot as plt

env = Game()

actions = list(range(5))
print(actions)

n = 21

start = time.time()

for i in range(n):
    action = random.choice(actions)
    state, score, done, d = env.step(action, update_score=i%5)
    print(f'{i} -- action: {action}, reward: {score}, done: {done}, dict: {d}')
    if i%10==0:
        plt.imshow(state)
        plt.title(f'frame: {i} -- action: {action}, reward: {score}, done: {done}')
        plt.show()

stop = time.time()
print(state.shape)

print('elapsed time: {:.2f}, fps = {:.2f}'.format(stop-start, n/(stop-start)))

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

env.close()
print('end')
