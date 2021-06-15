# QWOP Game Environment
# based on openai.gym environments structure

import pygetwindow
import pyautogui
import numpy as np
import cv2
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
GAME_HOFFSET = 10  # 220  # 10
GAME_VOFFSET = 174
GAME_WIDTH = 800  # 360  # 800
GAME_HEIGHT = 380  # 360  # 380
CHANNELS = 3
PAUSE_X = FULL_HOFFSET + 10
PAUSE_Y = FULL_VOFFSET + 10
MUTE_X  = FULL_HOFFSET + GAME_WIDTH - 10
# actions LUT - noop and 4 keys
# noop is represented as 'n' to ensure that each action lasts equally long
ACTION_LOOKUP = {
    0 : 'n',
    1 : 'q',
    2 : 'w',
    3 : 'o',
    4 : 'p'
}

class Game:

    def __init__(self, game_path=GAME_PATH):
        # game initialization
        self.sct = mss.mss()
        self.window = self.open_window(game_path)
        self.calculate_bboxes()
        
        self.game_steps = 0
        self.paused = False
        self.last_paused_ts = time.time()
        self.paused_time = 0
        self.unpaused_time = 0

        self.activate_game()
        self.mute()
        self.pause()
        self.last_score = 0

        self.action_space = len(ACTION_LOOKUP)
        self.state_space = (GAME_HEIGHT, GAME_WIDTH, CHANNELS)


    def step(self, action, update_score=False):
        # one step in agent-env information loop
        # returns: observation, reward, done, info_dict
        # game is unpaused only when taking an action
        self.unpause()
        self.take_action(action)
        game_shot, score_shot = self.take_screen_shots()
        self.pause()
        full_game = np.array(game_shot, dtype=np.uint8)[:,:,:CHANNELS]
        game_state = cv2.resize(full_game[:360,210:570,:], (0,0), fx=0.25, fy=0.25)
        done = self.is_done(full_game)
        if update_score or done:
            score = self.get_score(score_shot)
            self.last_score = score 
        else:
            score = self.last_score
        self.game_steps += 1
        # reward = self.calculate_reward(score)
        reward = score
        info_dict = {
            'unpaused_time': self.unpaused_time,
            'paused_time': self.paused_time
        }
        return game_state, reward, done, info_dict
    
    def step_test(self, action, update_score=False):
        self.unpause()
        self.take_action(action)
        game_shot, score_shot = self.take_screen_shots()
        # self.pause()
        full_game = np.array(game_shot, dtype=np.uint8)[:,:,:CHANNELS]
        game_state = cv2.resize(full_game[:360,210:570,:], (0,0), fx=0.25, fy=0.25)
        done = self.is_done(full_game)
        if update_score or done:
            score = self.get_score(score_shot)
            self.last_score = score 
        else:
            score = self.last_score
        self.game_steps += 1
        # reward = self.calculate_reward(score)
        reward = score
        info_dict = {
            'type': "test loop"
        }
        return game_state, reward, done, info_dict

    def reset(self):
        # uses qwop reset function
        # works even if player died
        pyautogui.keyDown('r')
        time.sleep(PRESS_TIME)
        pyautogui.keyUp('r')
        self.pause()
        self.game_steps = -1
        return self.step(0, update_score=True)

    def close(self):
        # only closes game window
        self.window.close()

    def render(self, mode='human'):
        # is it useful?
        pass

    def pause(self):
        # click '?' button in top left corner
        if not self.paused:
            pyautogui.click(self.q_button)
            self.unpaused_time = time.time() - self.last_paused_ts
            self.last_paused_ts += self.unpaused_time
            self.paused = True
            # print(f'paused on {self.last_paused_ts}')


    def unpause(self):
        # click '?' button in top left corner
        if self.paused:
            pyautogui.click(self.q_button)
            self.paused_time = time.time() - self.last_paused_ts
            self.last_paused_ts += self.paused_time
            self.paused = False
            # print(f'unpaused on {self.last_paused_ts}')

    
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
        self.q_button = (self.window.left+PAUSE_X, self.window.top+PAUSE_Y)
        self.mute_button = (self.window.left+MUTE_X, self.window.top+PAUSE_Y)

    def take_action(self, action_index):
        action = ACTION_LOOKUP[action_index]
        pyautogui.keyDown(action)
        time.sleep(PRESS_TIME)
        pyautogui.keyUp(action)
        
    def take_screen_shots(self):
        # returns both game and score BGRA screenshots
        game_shot = self.sct.grab(self.game_mon)
        score_shot = self.sct.grab(self.score_mon)
        return game_shot, score_shot

    def get_score(self, shot):
        img = np.array(shot)[:,:,:CHANNELS]  # BGR channels
        score = pytesseract.image_to_string(image=img)
        regex = re.compile("-?[0-9]+.?[0-9]")
        result = regex.findall(score)
        if len(result) > 0:
            score = result[0]
        else:
            score = 0
        return float(score)

    def is_done(self, game_shot):
        # yellow in BGR
        value = [0, 255, 255]
        coords = (46, 195)
        return all(game_shot[coords] == value)

    def calculate_reward(self, score):
        # TODO: define reward function
        return score
    
    def activate_game(self):
        self.window.activate()
        time.sleep(0.5)
        pyautogui.click(self.window.center)
        time.sleep(0.2)
        # self.pause()
    
    def mute(self):
        # click mute button in top right corner
        pyautogui.click(self.mute_button)

