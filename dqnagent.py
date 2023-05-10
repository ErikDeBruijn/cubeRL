import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from rubikscubenet import RubiksCubeNet
from environment import RubiksCubeEnv, ACTIONS
import os

MODEL_PATH = "saved_model-" + str(len(ACTIONS)) + "-actions.pt"

inverse_actions = {
    0: 1,  # R -> Ri
    1: 0,  # Ri -> R
    2: 3,  # L -> Li
    3: 2,  # Li -> L
    4: 5,  # U -> Ui
    5: 4,  # Ui -> U
    6: 7,  # D -> Di
    7: 6,  # Di -> D
    8: 9,  # F -> Fi
    9: 8,  # Fi -> F
    10: 11,  # B -> Bi
    11: 10  # Bi -> B
}


class DQNAgent:

    def __init__(self, input_size, action_space, device):
        output_size = action_space.n
        self.device = device
        self.action_space = action_space
        self.previous_action = None
        self.model = RubiksCubeNet(input_size, output_size).to(device)
        self.target_model = RubiksCubeNet(input_size, output_size).to(device)

        if os.path.exists(MODEL_PATH):
            print("Loading model from file: ", MODEL_PATH)
            self.model.load_state_dict(torch.load(MODEL_PATH))
            self.target_model.load_state_dict(torch.load(MODEL_PATH))
            self.model.to(device)
            self.print_model_weights()
        else:
            print("Creating new model")
            self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=10000)  # Experience replay buffer

        # If gamma is close to 0, the agent will primarily focus on immediate rewards, making it short-sighted and potentially leading to suboptimal policies.
        # If gamma is close to 1, the agent will consider future rewards more heavily, making it far-sighted and encouraging long-term planning.
        self.gamma = 0.5  # Discount factor (tried .99)
        self.epsilon = 0.01  # Exploration-exploitation factor
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.01
        self.batch_size = 64

    def act(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.model(state)
                valid_q_values = q_values.clone()

                if self.previous_action is not None:
                    inverse_action_idx = inverse_actions[self.previous_action]
                    valid_q_values[inverse_action_idx] = float('-inf')

                action_idx = torch.argmax(valid_q_values).item()
                self.previous_action = action_idx

                return action_idx
        else:
            action_idx = random.choice(range(self.action_space.n))

            if self.previous_action is not None:
                while action_idx == inverse_actions[self.previous_action]:
                    action_idx = random.choice(range(self.action_space.n))

            self.previous_action = action_idx

            return action_idx

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def print_model_weights(self):
        for name, param in self.model.named_parameters():
            print(name, param.data)
