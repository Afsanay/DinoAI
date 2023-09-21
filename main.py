# MSS is for screen capture
from mss import mss
# Sending commands
import pydirectinput
# Frame Processing
import cv2
# Transformation framework
import numpy as np
# OCR for game over extraction
import pytesseract
# Visualize Capture
from matplotlib import pyplot as plt
# Bring Time
import time
# Environment Components
import gym
from gym import Env
from gym.spaces import Box, Discrete
import torch
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


class WebGame(Env):
    # Setup the environment
    def __init__(self):
        # Subclass model
        super(WebGame, self).__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Define extraction
        self.cap = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 405, 'left': 630, 'width': 660, 'height': 70}

    # To do something
    def step(self, action):
        # Action key -> 0 = Space, 1= Duck(down), 2 = No Action
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action != 2:
            pydirectinput.press(action_map[action])

        # Checking if game is done
        res, done, done_cap = self.get_done()
        # Get new observation
        new_observation = self.get_observation()
        # Reward
        reward = 1
        # Info dict
        info = {}

        return new_observation, reward, done, info

    # Visualise the game
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    # Closes the observation
    def close(self):
        cv2.destroyAllWindows()

    # Restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()

    # Get the part of the observation that we want
    def get_observation(self):
        # Get screen capture
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        # GrayScale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, (100, 83))
        # Add Channels
        channel = np.reshape(resized, (1, 83, 100))
        return channel

    # Get done using OCR
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        # Valid done text
        done_strings = ['GAME', 'GAHE']
        # Apply OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return res, done, done_cap


env = WebGame()
# obs = env.get_observation()
# res, done, done_cap = env.get_done()
# plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB))
# plt.show()
# plt.imshow(done_cap)
# plt.show()
# print(res)
# print(done)

# play 10 games
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
    print(f'Total reward for episode {episode}: {total_reward}\n')

print(env.action_space.sample())
print(env.observation_space.sample())

env_checker.check_env(env)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            return True


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=500, save_path=CHECKPOINT_DIR)

model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR,
            verbose=1, buffer_size=12000, learning_starts=500)

model.learn(total_timesteps=1000, callback=callback)

model.load(os.path.join('./train/best_model_1000.zip'))

for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
    print(f'Total reward for episode {episode}: {total_reward}\n')
    time.sleep(2)


