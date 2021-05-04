# script to take a screenshot when needed

import pygetwindow as pgw 
import mss 
import time

HOFFSET = 10
VOFFSET = 64
WIDTH = 800
HEIGHT = 500

# screeen shot handler
sct = mss.mss()

# game window
game_title = "Adobe Flash Player 32"
windows = pgw.getWindowsWithTitle(game_title)
if len(windows) == 0:
    print('start the game first')
    exit()
game_window = windows[0]
# window activate
game_window.activate()
time.sleep(0.5)

# bbox
bbox = {"top": game_window.top + VOFFSET, "left": game_window.left + HOFFSET, "width": WIDTH, "height": HEIGHT}
# screen shot
sct_img = sct.grab(bbox)
# save
mss.tools.to_png(sct_img.rgb, sct_img.size, output="images/output.png")
print('ss saved to images/output.png')
