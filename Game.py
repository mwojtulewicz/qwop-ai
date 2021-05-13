# QWOP Game Environment
# based on openai.gym environments structure

import pygetwindow
import pyautogui
import numpy as np
import time
import mss 
import os
import re
import pytesseract

# local settings
from LocalSettings import LocalSettings
pytesseract.pytesseract.tesseract_cmd = LocalSettings.PYTESSERACT_EXE
GAME_PATH = LocalSettings.GAME_PATH

# constants
# general
PRESS_TIME = 0.08
WIN_TITLE = 'Adobe Flash Player 32'
# screenshots area bboxes
FULL_HOFFSET = 10
FULL_VOFFSET = 64
FULL_WIDTH = 800
FULL_HEIGHT = 500
SCORE_HOFFSET = 250
SCORE_VOFFSET = 89
SCORE_WIDTH = 320
SCORE_HEIGHT = 40
GAME_HOFFSET = 10
GAME_VOFFSET = 174
GAME_WIDTH = 800
GAME_HEIGHT = 380
PAUSE_X = FULL_HOFFSET + 10
PAUSE_Y = FULL_VOFFSET + 10
# actions LUT - 9 legal key combinations
# can be used in both basic (5 first actions) and full versions
ACTION_LOOKUP = {
    0 : '',
    1 : 'q',
    2 : 'w',
    3 : 'o',
    4 : 'p',
    5 : 'qo',
    6 : 'qp',
    7 : 'wo',
    8 : 'wp'
}

class Game:

    def __init__(self, game_path=GAME_PATH):
        # game initialization
        self.sct = mss.mss()
        self.window = self.open_window(game_path)
        self.calculate_bboxes()
        
        self.game_steps = 0
        self.paused = False

        self.activate_game()
        # self.pause()


    def step(self, action):
        # one step in agent-env information loop
        # returns: observation, reward, done, info_dict
        # maybe: unpause and pause
        # self.unpause()
        self.take_action(action)
        game_shot, score_shot = self.take_screen_shots()
        # self.pause()
        game_state = np.array(game_shot, dtype=np.uint8)[:,:,:3]
        score = self.get_score(score_shot)
        self.game_steps += 1
        done = self.is_done(game_state)
        reward = self.calculate_reward(score)
        # reward = (score, self.game_steps)
        return game_state, reward, done, {}

    def reset(self):
        # uses qwop reset function
        # works even if player died
        pyautogui.keyDown('r')
        time.sleep(PRESS_TIME)
        pyautogui.keyUp('r')
        # self.pause()
        self.game_steps = 0

    def close(self):
        # is it useful?
        pass

    def render(self, mode='human'):
        # is it useful?
        pass

    def pause(self):
        # click '?' button in top left corner
        if not self.paused:
            q_button = (self.window.left+PAUSE_X, self.window.top+PAUSE_Y)
            pyautogui.click(q_button)
            self.paused = True

    def unpause(self):
        # click '?' button in top left corner
        if self.paused:
            q_button = (self.window.left+PAUSE_X, self.window.top+PAUSE_Y)
            pyautogui.click(q_button)
            self.paused = False
    
    def open_window(self, game_path):
        # close all game windows
        for win in pygetwindow.getWindowsWithTitle(WIN_TITLE):
            win.close()
        # start a new one
        command = f'START /B {game_path}/flash.exe {game_path}/athletics.swf'
        os.system(command)
        # waiting for the window to open
        while not pygetwindow.getWindowsWithTitle(WIN_TITLE):
            time.sleep(0.1)
        # return game window
        return pygetwindow.getWindowsWithTitle(WIN_TITLE)[0]

    def calculate_bboxes(self):
        self.full_mon = {
            "top": self.window.top+FULL_VOFFSET, 
            "left": self.window.left+FULL_HOFFSET, 
            "width": FULL_WIDTH,
            "height": FULL_HEIGHT}
        self.game_mon = {
            "top": self.window.top+GAME_VOFFSET, 
            "left": self.window.left+GAME_HOFFSET, 
            "width": GAME_WIDTH,
            "height": GAME_HEIGHT}
        self.score_mon = {
            "top": self.window.top+SCORE_VOFFSET, 
            "left": self.window.left+SCORE_HOFFSET, 
            "width": SCORE_WIDTH,
            "height": SCORE_HEIGHT}

    def take_action(self, action):
        actions = ACTION_LOOKUP[action]
        for c in actions:
            pyautogui.keyDown(c)
        time.sleep(PRESS_TIME)
        for c in actions:
            pyautogui.keyUp(c)

    def take_screen_shots(self):
        # returns both game and score BGRA screenshots
        game_shot = self.sct.grab(self.game_mon)
        score_shot = self.sct.grab(self.score_mon)
        return game_shot, score_shot

    def get_score(self, shot):
        img = np.array(shot)[:,:,:3]  # takes 1 channel and invertes it
        score = pytesseract.image_to_string(image=img)
        regex = re.compile("-?[0-9]+.?[0-9]")
        result = regex.findall(score)
        if len(result) > 0:
            score = result[0]
        else:
            score = 0
        return float(score)

    def is_done(self, game_shot):
        # TODO: check if shot shows final screen
        # raise NotImplementedError
        return False

    def calculate_reward(self, score):
        # TODO: define reward function
        return score
    
    def activate_game(self):
        self.window.activate()
        time.sleep(0.5)
        pyautogui.click(self.window.center)
        time.sleep(0.2)
        # self.pause()







