import gym
from gym import spaces
from rubik.cube import Cube

from evaluation import score_cube, PATTERNS
import numpy as np
from lib import *

ACTIONS = ["R", "Ri", "L", "Li", "U", "Ui", "D", "Di", "F", "Fi", "B", "Bi",
           # "Y", "Yi",
           # "Li Ui L U F U Fi",  # swap out front left corner of second layer
           # "R U Ri Ui Fi Ui F"  # swap out front right corner of second layer
           ]


class RubiksCubeEnv(gym.Env):

    def __init__(self, cube):
        super(RubiksCubeEnv, self).__init__()

        self.cube = cube
        self.step_count = 0
        self.action_history = ""
        self.action_space = spaces.Discrete(len(ACTIONS))  # 6 faces * 2 directions (clockwise, counter-clockwise)
        # 9 stickers per face * 6 faces * 6 colors = 54
        self.observation_space = spaces.Box(low=0, high=1, shape=(54,),
                                            dtype=np.float32)  # 54 stickers, each with one of 6 colors

    def step(self, action):
        self.cube.sequence(ACTIONS[action])
        self.action_history = self.action_history + " " + ACTIONS[action]
        self.step_count += 1

        observation = self.get_observation()
        step_cost = 40
        reward = score_cube(self.cube) - step_cost
        done = self.is_solved()
        if done:
            print(f"{COLORS['green']} !! SOLVED in {self.step_count} turns !! Reward: {reward}{COLORS['reset']} (Scramble: {self.scrable}, sol: {self.action_history})")

        return observation, reward, done, {}

    def reset(self, permutations = 2):
        self.step_count = 0
        self.action_history = ""
        self.cube = Cube(PATTERNS["all"])
        # print("Initial cube:")
        # print_colored(self.cube)
        # Shuffle everything:
        self.scrable = ""
        for i in range(permutations):
            self.scrable = self.scrable + " " + np.random.choice(ACTIONS)
        self.cube.sequence(self.scrable)

        # print("Scrambled cube:")
        # print_colored(self.cube)

        return self.get_observation()

    def get_observation(self):
        color_mapping = {'R': 0, 'G': 1, 'B': 2, 'Y': 3, 'W': 4, 'O': 5}

        # convert string to array:
        state = self.cube.flat_str()
        # convert into numeric values
        cube_array = np.array([color_mapping[color] for color in state])

        return cube_array

    def is_solved(self):
        return str(PATTERNS["all"]) == str(self.cube)
