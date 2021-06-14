import numpy as np
import matplotlib.pyplot as plt

MAX_EPS = 1
MIN_EPS = 0.1
DECAY = 0.01

def epsilon(episode):
    return MIN_EPS + (MAX_EPS-MIN_EPS)*np.exp(-DECAY*episode)

x = np.arange(0,500)
y = epsilon(x)

plt.figure(figsize=(12,6.5))
plt.title('Epsilon decay'); plt.xlabel('Episode number'); plt.ylabel('Value')
plt.plot(x, y)
plt.ylim((-0.1,1.1)); plt.grid()
plt.tight_layout()
# plt.savefig('models/eps_decay')
plt.show()