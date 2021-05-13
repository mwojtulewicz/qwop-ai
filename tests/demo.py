import os 
import pygetwindow as pgw 
import mss 
import time
import pyautogui


# screeen shot handler
sct = mss.mss()

# ----------------------------------------------------------------------------
# starting the game and extracting its window

# all opened windows
windows_before = pgw.getAllTitles()
print('windows before : ', len(windows_before))

# starting the game (on Windows in detached mode)
game_path = '../game'
command = f'START /B {game_path}/flash.exe {game_path}/athletics.swf'
print('starting game...')
os.system(command)
time.sleep(1)

# all opened windows
windows_after = pgw.getAllTitles()
print('windows : ', len(windows_after))

# extracting game window title
game_title = [x for x in windows_after if x not in windows_before]
game_title = ''.join(game_title)
print('game window title : ', game_title)
time.sleep(1)

# firsst element
game_window = pgw.getWindowsWithTitle(game_title)[0]
print('qwop window : ', game_window)


# ----------------------------------------------------------------------------
# control demo

# activating the window
game_window.activate()

# click the window's center to start the game
center = (game_window.left + game_window.width/2, game_window.top + game_window.height/2)
pyautogui.click(center)

# testing actions
for key in 'qwop':
    print(f'pressing {key}...')
    pyautogui.keyDown(key)
    time.sleep(0.08)
    pyautogui.keyUp(key)

# pausing test
q_button = (game_window.left + 10 + 10, game_window.top + 64 + 10)
pyautogui.click(q_button)
print('pausing...')
time.sleep(2)
pyautogui.click(q_button)

# -----------------------------------------------------------------------------
# screen shot

print('saving screen shots...')

# game
game_bbox = {"top": game_window.top + 174, "left": game_window.left + 10, "width": 800, "height": 380}
# screen shot
sct_img = sct.grab(game_bbox)
# save
mss.tools.to_png(sct_img.rgb, sct_img.size, output="images/game.png")

# score
score_bbox = {"top": game_window.top + 89, "left": game_window.left + 250, "width": 320, "height": 40}
# screen shot
sct_img = sct.grab(score_bbox)
# save
mss.tools.to_png(sct_img.rgb, sct_img.size, output="images/score.png")


# ----------------------------------------------------------------------------
# closing the game window
game_window.close()
print('done')
