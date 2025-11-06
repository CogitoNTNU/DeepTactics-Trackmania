"""
Training script for TrackMania (TMRL) environment
Uses Rainbow agent with convolutional layers and car features
"""
import os
import torch
from src.helper_functions.tm_checkpointing import cleanup_old_checkpoints, resume_from_checkpoint, setup_checkpoint_dir
import wandb
from src.agents.rainbow_tm import Rainbow
from tensordict import TensorDict
from config_files.tm_config import Config_tm
from src.helper_functions.tm_actions import map_action_tm
from sys import platform
import numpy as np
if platform != 'darwin' and platform != 'linux':
    from tmrl import get_environment

def run_training():
    # Load .env file from project root
    config = Config_tm()
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    rainbow_agent = Rainbow(config)

    features = []
    if config.use_dueling:
        features.append("Dueling")
    if config.use_prioritized_replay:
        features.append("PER")
    if config.use_doubleDQN:
        features.append("Double")
    
    feature_str = "+".join(features) if features else "Basic"
    run_name = f"{config.run_name}_{rainbow_agent.__class__.__name__}_{config.rtgym_interface}_{feature_str}"

    if config.checkpoint:
        if config.load_checkpoint:
            checkpoint_dir = os.path.join(config.checkpoint_dir, config.load_checkpoint_name)
            checkpoint_dir = setup_checkpoint_dir(checkpoint_dir)
        else:
            checkpoint_dir = os.path.join(config.checkpoint_dir, run_name)
            checkpoint_dir = setup_checkpoint_dir(checkpoint_dir)


    print(f"Training on device: {rainbow_agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Try to load from checkpoint if resume is enabled
    if config.resume_from_checkpoint:
        start_episode, start_step, wandb_run_id = resume_from_checkpoint(rainbow_agent, checkpoint_dir)
    else:
        start_episode = 0
        start_step = 0
        wandb_run_id = None

    #act_buf_len
    act_buf_len = config.act_buf_len

    # Get TrackMania environment
    env = get_environment()
    
    # Resume WandB run if we have a run_id, otherwise create new
    if wandb_run_id:
        run_context = wandb.init(project="Trackmania", name=run_name, id=wandb_run_id, resume="allow", config=rainbow_agent.config)
    else:
        run_context = wandb.init(project="Trackmania", name=run_name, config=rainbow_agent.config)

    with run_context as run:
        run.watch(rainbow_agent.policy_network, log="all", log_freq=100)
        run.watch(rainbow_agent.target_network, log="all", log_freq=100)

        tot_reward = 0
        episode = start_episode
        tot_q_value = 0
        n_q_values = 0
        max_steps = config.ep_max_length
        crash_threshold = config.crash_threshold
        crash_detection = config.crash_detection
        crash_penalty = config.crash_penalty
        #for TM20FULL obs = [velocity, gear, rpm, images]
        #images greyscale obs[3].shape >>> (IMG_HIST_LEN,x,y)
        #images full color obs[3].shape >>> (IMG_HIST_LEN,x,y,rgb(3))
        #velocity type(obs[0]) >>> array(1,) || velocity type(obs[0][0]) >> numpy.float32
        #gear type(obs[1]) >>> array(1,) || gear type(obs[1][0]) >> numpy.float32
        #rpm type(obs[2]) >>> array(1,) || rpm type(obs[2][0]) >> numpy.float32
        #example obs:[001.4, 0.0, 01772.1, imgs(1)] obs:[028.0, 2.0, 06113.0, imgs(1)]
        episode_step = 0
        observation, info = env.reset()

        try:
            for i in range(start_step, config.training_steps):
    
                image_tensor = torch.tensor(observation[3], dtype=torch.float32)/255
                # image_tensor = torch.tensor(observation[3][0], dtype=torch.float32)/255
                # image_tensor = image_tensor.unsqueeze(0)
                # image_tensor = image_tensor.permute(2, 0, 1) # for color images

                car_features = torch.tensor([observation[0][0], observation[1][0], observation[2][0]], dtype=torch.float32)
                
                action_history = []
                for j in range(4, 4 + act_buf_len):
                    action_history.extend(observation[j])
                action_history = torch.tensor(action_history, dtype=torch.float32)

                action, q_value = rainbow_agent.get_action(image_tensor.unsqueeze(0), car_features.unsqueeze(0), action_history.unsqueeze(0))
                # print(f"Action: {action}")
                # Ensure action is a plain int (agent might return a tensor)
                if hasattr(action, "item"):
                    action_idx = int(action.item())
                else:
                    action_idx = int(action)

                # Map discrete action index terminated-> Trackmania control vector [steer, accel, brake]
                mapped_action = map_action_tm(action_idx)

                if q_value is not None:
                    tot_q_value += q_value
                    n_q_values += 1

                
                next_obs, reward, terminated, truncated, info = env.step(mapped_action)
                
                episode_step += 1
                done = terminated or truncated
                
                if crash_detection and observation[0][0] - next_obs[0][0] > crash_threshold:
                    reward -= crash_penalty 
                
                if info['reached_finishline']:
                    speed_ratio = max_steps / episode_step
                    time_bonus = min(500, 100 * np.exp(speed_ratio - 1))
                    reward += time_bonus

                # Process next observation tensors
                next_image_tensor = torch.tensor(next_obs[3], dtype=torch.float32)/255 #for batched images remember to update conv batch size
                # next_image_tensor = torch.tensor(next_obs[3][0], dtype=torch.float32) / 255 #for singel image remember to update conv batch size
                # next_image_tensor = next_image_tensor.unsqueeze(0) #for singel image
                # next_image_tensor = next_image_tensor.permute(2, 0, 1) # for singel color image

                # Next car features (current state only)
                next_car_features = torch.tensor([next_obs[0][0], next_obs[1][0], next_obs[2][0]], dtype=torch.float32)
                
                # Next action history
                next_action_history = []
                for j in range(4, 4 + act_buf_len):
                    next_action_history.extend(next_obs[j])
                next_action_history = torch.tensor(next_action_history, dtype=torch.float32)


                experience = TensorDict({
                    "image": image_tensor,
                    "car_features": car_features,
                    "action_history": action_history,
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_image": next_image_tensor,
                    "next_car_features": next_car_features,
                    "next_action_history": next_action_history,
                    "done": torch.tensor(done)
                }, batch_size=torch.Size([]))

                rainbow_agent.store_transition(experience)
                tot_reward += float(reward)

                loss = rainbow_agent.train()

                # update model per step instead of per epsiode we have soft actor now
                if i % config.target_network_update_frequency == 0:
                    rainbow_agent.update_target_network()

                race_complete_time = 0
                if done:
                    if n_q_values > 0:
                        avg_q_value = tot_q_value / n_q_values
                    else:
                        avg_q_value = -1
                    if  info['reached_finishline']:
                        race_complete_time = episode_step * config.time_step_duration
                        
                        
                    log_metrics = {
                        "episode_reward": tot_reward,
                        "loss": loss,
                        "learning_rate": rainbow_agent.optimizer.param_groups[0]['lr'],
                        "q_values": avg_q_value,
                        "epsilon": rainbow_agent.epsilon,
                        "race_complete_time" : race_complete_time 
                    }

                    rainbow_agent.decay_epsilon(episode)

                    run.log(log_metrics, step=episode)

                    episode += 1
                    tot_reward = 0
                    tot_q_value = 0
                    n_q_values = 0
                    episode_step = 0

                    # Save checkpoint every N episodes
                    if episode % config.checkpoint_frequency == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pt")
                        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

                        rainbow_agent.save_checkpoint(checkpoint_path, episode, i, {"run_id": run.id})
                        rainbow_agent.save_checkpoint(latest_path, episode, i, {"run_id": run.id})

                        # Clean up old checkpoints
                        cleanup_old_checkpoints(checkpoint_dir, config.keep_last_n_checkpoints)

                    observation, info = env.reset()
                else:
                    observation = next_obs

                # if i % config.target_network_update_frequency == 0:
                #     rainbow_agent.update_target_network()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            crash_path = os.path.join(checkpoint_dir, "checkpoint_interrupted.pt")
            print(f"Saving checkpoint due to interruption...")
            rainbow_agent.save_checkpoint(crash_path, episode, i, {"run_id": run.id, "reason": "user_interrupt"})
            raise

        except Exception as e:
            print(f"\nTraining crashed with error: {e}")
            crash_path = os.path.join(checkpoint_dir, "checkpoint_crash.pt")
            print(f"Saving emergency checkpoint...")
            try:
                rainbow_agent.save_checkpoint(crash_path, episode, i, {"run_id": run.id, "error": str(e)})
            except Exception as save_error:
                print(f"Failed to save crash checkpoint: {save_error}")
            raise

        finally:
            # Always save a final checkpoint when training ends
            final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
            print(f"Saving final checkpoint...")
            try:
                rainbow_agent.save_checkpoint(final_path, episode, i, {"run_id": run.id})
            except Exception as save_error:
                print(f"Failed to save final checkpoint: {save_error}")

    env.close()

if __name__ == "__main__":
    run_training()