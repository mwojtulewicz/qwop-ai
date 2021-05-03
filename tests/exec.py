import os 
import pygetwindow as pgw 
import mss 
import time


# screeen shot handler
sct = mss.mss()

# ----------------------------------------------------------------------------
# starting the game and extracting its windows coords

windows_before = pgw.getAllTitles()
print('windows before : ', len(windows_before))

print('starting game...')
game_path = 'C:/Users/mateu/Documents/studia/6semestr/sp/source/game'
command = f'START /B {game_path}/flash.exe {game_path}/athletics.swf'
os.system(command)
time.sleep(1)

windows_after = pgw.getAllTitles()
print('windows : ', len(windows_after))

game_title = [x for x in windows_after if x not in windows_before]
game_title = ''.join(game_title)
print('game window title : ', game_title)
time.sleep(1)

game_window = pgw.getWindowsWithTitle(game_title)[0]
print('qwop window : ', game_window)

# activating the window
game_window.activate()

# -----------------------------------------------------------------------------
# screen shot

# The screen part to capture 
monitor = {"top": game_window.top, "left": game_window.left, "width": game_window.width, "height": game_window.height}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

# getting screen shot
sct_img = sct.grab(monitor)

# Save to the picture file
mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
print('output file : ', output)

