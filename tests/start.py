# script to start the game

import os

GAME_PATH = 'C:/Users/mateu/Documents/studia/6semestr/sp/source/game'

# starting the game in detached mode on Windows
command = f'START /B {GAME_PATH}/flash.exe {GAME_PATH}/athletics.swf'
print('starting...')
os.system(command)
