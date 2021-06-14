import numpy as np

x = np.ones(shape=(3,3))
y = np.zeros(shape=(3,3))

z = np.stack((x,y),axis=2)
print(z.shape, z)