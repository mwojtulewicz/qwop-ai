import numpy as np

x = np.ones(shape=(1,3,3))
y = np.zeros(shape=(1,3,3))

z = np.stack((x,y),axis=-1)
print(z.shape, z)