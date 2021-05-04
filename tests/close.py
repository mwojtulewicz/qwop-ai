# script that closes every game window

import pygetwindow
import time

title = 'Adobe Flash Player 32'
for window in pygetwindow.getWindowsWithTitle(title):
    # window.activate()  # raises error, dunno why
    window.close()
