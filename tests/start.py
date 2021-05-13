# script to start the game

import os


def start():
    GAME_PATH = '../game'

    # starting the game in detached mode on Windows
    command = f'START /B {GAME_PATH}/flash.exe {GAME_PATH}/athletics.swf'
    print('starting...')
    os.system(command)

if __name__=='__main__':
    start()