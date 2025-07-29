import os
import sys
# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import time
from datetime import datetime

import torch
import numpy as np

from endpoints.training_interface import TrainInterface
from endpoints.data_parser import DataParser
from gameplay.scorekeeper import ScoreKeeper
from models.PPO import PPO


def create_environment():
    """Create the training environment"""
    from endpoints.training_interface import TrainInterface
    
    # Use the correct status model
    env = TrainInterface(classifier_model_file='models/transfer_status_baseline.pth')
    return env


def train():
    """Main training function"""
    print("=" * 80)
    print("Starting RL Training for Zombie Apocalypse Decision Game")
    print("=" * 80)
    
    # Create environment with correct model
    env = TrainInterface(classifier_model_file='models/transfer_status_baseline.pth')

    env_name = "ZombieRL"
    
    has_continuous_action_space = False  # discrete action space (6 actions: SKIP_BOTH, SQUISH_LEFT, SQUISH_RIGHT, SAVE_LEFT, SAVE_RIGHT, SCRAM)

    max_ep_len = 200                    # max timesteps in one episode (increased for better learning)
    max_training_timesteps = int(5e5)   # 500K timesteps = ~day/week of training for 450+ scores

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps) - more frequent for long runs

    # PPO hyperparameters
    update_timestep = max_ep_len * 2    # update policy every n timesteps
    K_epochs = 40                       # update policy for K epochs in one PPO update
    eps_clip = 0.2                      # clip parameter for PPO
    gamma = 0.99                        # discount factor
    lr_actor = 0.0003                   # learning rate for actor network
    lr_critic = 0.001                   # learning rate for critic network

    random_seed = 42                    # set random seed for reproducibility

    # Logging setup
    log_dir = "model_training"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    # Get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # Create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    # Checkpointing setup - use same run_num as logs for unique model names
    directory = "model_training"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    # Use run_num (not hardcoded 0) to ensure unique model names for each training run
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num)
    print("save checkpoint path : " + checkpoint_path)

    # Print all hyperparameters
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("Action space size:", env.action_space.n)
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    print("============================================================================================")

    # Initialize PPO Agent
    ppo_agent = PPO(lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space)

    # Track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # Logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # Printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # Performance tracking for long training
    best_episode_reward = float('-inf')
    episodes_without_improvement = 0
    max_episodes_without_improvement = 1000  # Early stopping after 1000 episodes (allows for long exploration phases)
    
    # Training loop with auto-resume capability
    try:
        print(f"🚀 Starting long-term training (Target: {max_training_timesteps:,} timesteps)")
        print(f"⏱️  Expected duration: Several hours to days depending on performance")
        print(f"💾 Models saved every {save_model_freq:,} timesteps")
        print(f"🛑 Early stopping after {max_episodes_without_improvement} episodes without improvement (allows for long exploration)")
        print("=" * 80)
        
        while time_step <= max_training_timesteps:
            state = env.reset()
            current_ep_reward = 0

            for t in range(1, max_ep_len+1):
                # Select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, truncated, info = env.step(action)

                # Saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done or truncated)

                time_step += 1
                current_ep_reward += reward

                # Update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # Log in logging file
                if time_step % log_freq == 0:
                    # Log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # Printing average reward
                if time_step % print_freq == 0:
                    # Print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # Save model weights
                if time_step % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

                # Break if the episode is over
                if done or truncated:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            i_episode += 1
            
            # Long-term performance tracking
            episode_score = env.scorekeeper.get_final_score()
            
            # Track best performance for early stopping
            if current_ep_reward > best_episode_reward:
                best_episode_reward = current_ep_reward
                episodes_without_improvement = 0
                print(f"🏆 NEW BEST! Episode {i_episode-1}: Reward={current_ep_reward:.2f}, Score={episode_score:.1f}")
            else:
                episodes_without_improvement += 1
            
            # Print progress every 10 episodes
            if i_episode % 10 == 0:
                print(f"📊 Episode {i_episode-1}: Reward={current_ep_reward:.2f}, Score={episode_score:.1f}, Best={best_episode_reward:.2f}")
            
            # Early stopping check for very long runs
            if episodes_without_improvement >= max_episodes_without_improvement:
                print(f"🛑 Early stopping: No improvement for {max_episodes_without_improvement} episodes")
                print(f"Best reward achieved: {best_episode_reward:.2f}")
                break

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print(f"Best reward achieved before interruption: {best_episode_reward:.2f}")
    except Exception as e:
        print(f"❌ Training error: {e}")
        print(f"Last checkpoint saved at timestep: {time_step}")
        print("You can resume training by restarting with the same run number")
    finally:
        log_f.close()
        print(f"\n📈 Training completed {i_episode} episodes in {time_step} timesteps")

    # Print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    
    # Save final model with unique run number
    final_checkpoint_path = directory + "PPO_{}_FINAL_{}_{}.pth".format(env_name, random_seed, run_num)
    ppo_agent.save(final_checkpoint_path)
    print("Final model saved at:", final_checkpoint_path)
    print("============================================================================================")

    return ppo_agent, env


def test_environment():
    """
    Test function to verify the environment works correctly
    """
    print("Testing environment setup...")
    
    try:
        env = create_environment()
        state = env.reset()
        
        print("Environment created successfully!")
        print("Observation space keys:", list(state.keys()))
        print("Action space size:", env.action_space.n)
        print("Sample observation shapes:")
        for key, value in state.items():
            print(f"  {key}: {np.array(value).shape}")
        
        # Test a few random actions
        for i in range(5):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={done}")
            if done:
                state = env.reset()
                print("Episode finished, environment reset")
                
        print("Environment test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        return False


if __name__ == '__main__':
    print("Zombie Apocalypse RL Training")
    print("1. Testing environment...")
    
    if test_environment():
        print("2. Starting training...")
        train()
    else:
        print("Environment test failed. Please check the setup.")