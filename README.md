# Rubik's Cube Solver using Deep Reinforcement Learning

This project aims to solve the Rubik's Cube using Deep Reinforcement Learning. The solver is trained using a Deep Q-Network (DQN) and a custom environment for the Rubik's Cube.

## ToDo's
- [x] Create evaluation metrics (score_cube)
- [x] Create a set of actions
- [x] Convert observations into an array that the network can handle
- [x] Create a network
- [x] Output how the cube was scrambled
- [x] Only execute actions that aren't inversions of a previous turn (since the pair is useless)
- [x] When a new 'best' solution is found (partially solved or entirely), display the solution as a coloured cube
- [x] Display the actions taken (all turns taken) that led to a solution
- [ ] Figure out if the performance is 'better than random'
- [ ] 'Anihilate' cube turns in the action history if the moves are inversions of a previous move.

## Features

- Implemented in Python using the PyTorch library for deep learning
- Customizable Rubik's Cube environment
- DQN Agent that learns to solve the Rubik's Cube
- Prevents inverting previous actions during training
- Colorful terminal output to visualize the cube's state

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Gym
- [rubik-cube](https://github.com/pglass/cube) 

## Installation

1. Clone the repository:

`git clone git@github.com:ErikDeBruijn/cubeRL.git`


2. Install the required packages:

`pip install -r requirements.txt`

## Usage

1. Train the DQN Agent:

`python main.py`

The training process will display the progress, including the episode number, reward, highest reward, epsilon, and steps taken. The cube's state will be printed when the training is interrupted.

The model will be 'checkpointed' upon a signal.

2. (Optional) To load a pre-trained model and resume training, ensure the saved model file (`saved_model.pt`) is in the project directory and uncomment the corresponding lines in `main.py`.

## Customization

- Modify the DQN architecture in `dqnagent.py`
- Adjust the hyperparameters in `main.py` and `dqnagent.py`
- Customize the reward function in the Rubik's Cube environment

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
