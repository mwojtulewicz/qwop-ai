import numpy as np
from scipy.special import softmax 

x = [-100,2,3,49,50]
xs = softmax(x)

print(xs)

print(np.random.choice(5,p=xs))