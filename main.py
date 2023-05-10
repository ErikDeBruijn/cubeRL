from rubik.cube import Cube
from environment import RubiksCubeEnv
from dqnagent import DQNAgent, MODEL_PATH
import torch
from evaluation import *
from lib import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()

    cube = Cube(PATTERNS["all"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = RubiksCubeEnv(cube)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space, device)

    if args.verbose:
        print("Initial cube state:")
        print_colored(cube)
        print("Loaded model weights:")
        agent.print_model_weights()

    num_episodes = 200000
    update_target_frequency = 5
    steps_until_target_update = 0
    highest_reward_ever = 0
    scramble_permutations = 9.0
    solved_count = 0

    try:
        for episode in range(num_episodes):
            state = env.reset(int(scramble_permutations))
            done = False
            max_steps = int(scramble_permutations * 1.1 + 8)
            episode_reward = 0
            steps_taken = 0
            highest_reward = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                steps_taken += 1

                highest_reward = max(highest_reward, reward)
                if reward > (highest_reward + 4) and reward > 30:
                    highest_reward = reward
                    reward += 500

                if reward > highest_reward_ever:
                    highest_reward_ever = reward
                    reward += 500
                    print("\nNew highest reward:", highest_reward_ever, " at step: ", steps_taken)
                    print_colored(env.cube)
                    print("Score: ", score_cube(env.cube))
                reward = reward - steps_taken * 0.05  # - extra_penalty
                agent.remember(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                if steps_taken >= max_steps:
                    done = True
                steps_until_target_update += 1

                if steps_until_target_update % update_target_frequency == 0:
                    agent.update_target_model()

                agent.train()

                if env.is_solved():
                    solved_count += 1
                    if solved_count / (episode + 1) > 0.90:
                        scramble_permutations += 0.1
                    break

            print(
                f"\rEpisode: {episode}, solved: {solved_count / (episode + 1) * 100.0:.1f}%, Reward: {episode_reward:.1f}, HiReward: {highest_reward:.1f} Epsilon: {agent.epsilon:.4f} Steps: {steps_taken} Scrambled {scramble_permutations:.1f} turns.")
    except KeyboardInterrupt:
        print("Training interrupted.")
        print("Current cube state:")
        print_colored(env.cube)
        print("Saving model to: ", MODEL_PATH)
        torch.save(agent.model.state_dict(), MODEL_PATH)
