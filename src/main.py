import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agents.eval_agents import NashEquilibriumAgent
from agents.nfsp_agent import NFSPAgent
from games.kuhn_poker import KuhnPoker

# Configuration parameters (these should be adjusted based on your specific needs)
anticipatory_param = 0.1  # NFSP anticipatory parameter
epsilon_start = 1.0  # Starting value of epsilon for epsilon-greedy action selection
epsilon_end = 0.1  # Final value of epsilon
epsilon_decay = 0.995  # Factor to decay epsilon after each episode
batch_size = 256  # Batch size for training
num_episodes = 10000  # Number of episodes to train
eval_freq = 100  # Frequency of evaluation
eval_episodes = 10000  # Number of episodes to evaluate
target_update_interval = 100  # Frequency to update the target network

avg_reward_window_size = (
    100  # Window size for calculating the moving average reward (for plot)
)
save_freq = eval_freq  # Frequency to save the model
save_dir = "models"  # Directory to save models
save_prefix = "nfsp"  # Prefix for the saved models
plot_folder = "plots"  # Folder to save plots


def evaluate_agent(env, agent, num_episodes=10):
    total_rewards = 0
    eval_agent = NashEquilibriumAgent(position=1, action_space=env.action_space)
    for i in range(num_episodes):
        rl_turn = 0 if i % 2 == 0 else 1
        eval_agent.position = 1 - rl_turn
        state = env.reset()
        done = False
        while not done:
            action_rl = agent.select_action(
                state, epsilon=0
            )  # No exploration during evaluation
            action_opp = eval_agent.select_action(state)

            action = action_rl if env.current_player == rl_turn else action_opp
            next_state, reward, done, _ = env.step(action)
            state = next_state

            if done:
                total_rewards += reward
    average_reward = total_rewards / num_episodes
    return average_reward


def save_checkpoint(agent, save_dir):
    """Saves the state of the agent, including all networks and optimizers.

    Args:
        agent (NFSPAgent): The agent to save.
        save_dir (str): The file path to save the agent state.
    """
    path = save_dir.rsplit("/", 1)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {
        "br_net_state_dict": agent.br_net.state_dict(),
        "as_net_state_dict": agent.as_net.state_dict(),
        "br_optimizer_state_dict": agent.br_optimizer.state_dict(),
        "as_optimizer_state_dict": agent.as_optimizer.state_dict(),
        "target_br_net_state_dict": (
            agent.target_br_net.state_dict()
            if hasattr(agent, "target_br_net")
            else None
        ),
    }
    torch.save(checkpoint, save_dir)


def main():
    # Initialize the game environment
    env = KuhnPoker()  # Kuhn Poker

    # Initialize the NFSP Agent
    agent = NFSPAgent(env.state_space, env.action_space, anticipatory_param)

    # Initialize epsilon for epsilon-greedy action selection
    epsilon = epsilon_start

    # Initialization for plotting
    evaluation_scores = []
    average_rewards = []

    now = time.time()  # Get the current time for saving the files

    # Main training loop
    for episode in tqdm(range(1, num_episodes + 1)):
        # Reset the environment for a new episode
        state = env.reset()
        total_ep_rewards = 0  # Track rewards accumulated in the episode

        while True:  # Continue until the episode is done
            # Agent selects action based on current state
            action = agent.select_action(state, epsilon)
            if_best_response_used = agent.last_action_best_response

            # Environment executes action, returns next state and reward
            next_state, reward, done, _ = env.step(action)
            total_ep_rewards += reward

            # Agent stores experience in memory
            agent.store_experience_rl(state, action, reward, next_state, done)
            if if_best_response_used:
                agent.store_experience_sl(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Break the loop if the episode is done
            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Train the agent after each episode
        agent.train(batch_size)

        # Periodically update the target network
        if episode % target_update_interval == 0:
            agent.update_target_network()

        # Logging
        average_rewards.append(total_ep_rewards)

        if episode % eval_freq == 0:
            evaluation_score = evaluate_agent(env, agent, eval_episodes)
            evaluation_scores.append(evaluation_score)
            tqdm.write(f"Evaluation score after episode {episode}: {evaluation_score}")

        # Save the model
        if episode % save_freq == 0:
            save_checkpoint(
                agent, f"{save_dir}/{now}/{save_prefix}_episode_{episode}.pt"
            )
            tqdm.write("\033[1;32m" + f"Saved model at episode {episode}" + "\033[0m")

    rewards_series = pd.Series(average_rewards)

    # Calculate Exponential Moving Average
    ema_rewards = rewards_series.ewm(span=avg_reward_window_size).mean().values

    # Ensure the output directory exists
    output_dir = f"{plot_folder}/{now}"
    os.makedirs(output_dir, exist_ok=True)

    # Plotting the evaluation scores (no changes needed here)
    plt.figure(figsize=(10, 6))  # Optionally specify the figure size
    plt.plot(range(eval_freq, num_episodes + 1, eval_freq), evaluation_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("NFSP Agent Evaluation Scores")
    plt.savefig(f"{output_dir}/nfsp_evaluation_scores.png")
    plt.clf()  # Clear the figure for the next plot

    # Plotting the smoothed moving average rewards
    plt.figure(figsize=(10, 6))  # Optionally specify the figure size
    episodes_x = np.linspace(
        avg_reward_window_size, num_episodes, len(ema_rewards)
    )  # Adjust as needed
    plt.plot(episodes_x, ema_rewards)  # Use ema_rewards if using EMA
    plt.xlabel("Episodes")
    plt.ylabel(f"Moving Average Reward (Window={avg_reward_window_size})")
    plt.title("NFSP Agent Training - Moving Average Reward")
    plt.savefig(f"{output_dir}/nfsp_moving_avg_rewards.png")


if __name__ == "__main__":
    main()
