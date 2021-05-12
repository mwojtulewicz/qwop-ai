# just some testing

import pygetwindow
import time
import mss
from start import start
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# starting game 
start()

while len(pygetwindow.getWindowsWithTitle('Adobe Flash Player 32'))==0:
    pass

# mss object before screen grabbing
sct = mss.mss()

# window
window = pygetwindow.getWindowsWithTitle('Adobe Flash Player 32')[0]
# print(type(window))
# print(dir(window))
# print(window.size)

# bbox
FULL_HOFFSET = 10
FULL_VOFFSET = 64
FULL_WIDTH = 800
FULL_HEIGHT = 500
full_mon = {"top": window.top+FULL_VOFFSET, 
            "left": window.left+FULL_HOFFSET,
            "width": FULL_WIDTH,
            "height": FULL_HEIGHT}

SCORE_HOFFSET = 250
SCORE_VOFFSET = 89
SCORE_WIDTH = 320
SCORE_HEIGHT = 40
score_mon = {"top": window.top+SCORE_VOFFSET, 
            "left": window.left+SCORE_HOFFSET,
            "width": SCORE_WIDTH,
            "height": SCORE_HEIGHT}

GAME_HOFFSET = 10
GAME_VOFFSET = 174
GAME_WIDTH = 800
GAME_HEIGHT = 380
game_mon = {"top": window.top+GAME_VOFFSET, 
            "left": window.left+GAME_HOFFSET,
            "width": GAME_WIDTH,
            "height": GAME_HEIGHT}


def mss_rgb(im):
    """ 21fps """
    return np.asarray(im.rgb)


def numpy_flip(im):
    """ 29fps """
    frame = np.array(im, dtype=np.uint8)
    return np.flip(frame[:, :, :3], 2)


def numpy_slice(im):
    """ 21fps """
    return np.array(im, dtype=np.uint8)[..., [2, 1, 0]]


def pil_frombytes(im):
    """ 22fps """
    return np.asarray(Image.frombytes('RGB', im.size, im.bgra, 'raw', 'BGRX'))

def best_method(im):
    """ 
    - 30fps 
    - its BGRA 
    - can be sliced to 1 channel
    """
    return np.array(im, dtype=np.uint8)[:,:,:3]

fps = 0
last_time = time.time()
while time.time() - last_time < 1:
    out = best_method(sct.grab(game_mon))
    out2 = best_method(sct.grab(score_mon))
    fps += 1
print(fps)

print(out.shape)

plt.imshow(out, 'gray') 
plt.show()
