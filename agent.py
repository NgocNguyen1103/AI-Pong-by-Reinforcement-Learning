# from buffer import ReplayBuffer
# from model import Model, soft_update
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# import datetime
# import time
# from torch.utils.tensorboard import SummaryWriter
# import random
# import os
# import cv2


# class Agent():

#     def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):
        
#         self.env = env

#         self.step_repeat = step_repeat

#         self.gamma = gamma

#         obs, info = self.env.reset()

#         obs = self.process_observation(obs)

#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#         print(f'Loaded model on device {self.device}')

#         self.memory = ReplayBuffer(max_size=500000, input_shape=obs.shape, device=self.device)

#         self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        
#         self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)

#         self.target_model.load_state_dict(self.model.state_dict())

#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

#         self.learning_rate = learning_rate


#     def process_observation(self, obs):
#         obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
#         return obs


#     def test(self):

#         self.model.load_the_model()

#         obs, info = self.env.reset()

#         done = False
#         obs, info = self.env.reset()
#         obs = self.process_observation(obs)

#         episode_reward = 0

#         while not done:

#             if random.random() < 0.05:
#                 action = self.env.action_space.sample()
#             else:
#                 q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
#                 action = torch.argmax(q_values, dim=-1).item()
            
#             reward = 0

#             for i in range(self.step_repeat):
#                 reward_temp = 0

#                 next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

#                 reward += reward_temp

#                 frame = self.env.env.env.render() 

#                 resized_frame = cv2.resize(frame, (500, 400))

#                 resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

#                 cv2.imshow("Pong AI", resized_frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#                 time.sleep(0.05)

#                 if(done):
#                     break
            
#             obs = self.process_observation(next_obs)

#             episode_reward += reward


#     def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):

#         summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
#         writer = SummaryWriter(summary_writer_name)

#         if not os.path.exists('models'):
#             os.makedirs('models')

#         total_steps = 0

#         for episode in range(episodes):

#             done = False
#             episode_reward = 0
#             obs, info = self.env.reset()
#             obs = self.process_observation(obs)

#             episode_steps = 0

#             episode_start_time = time.time()

#             while not done and episode_steps < max_episode_steps:

#                 if random.random() < epsilon:
#                     action = self.env.action_space.sample()
#                 else:
#                     q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
#                     action = torch.argmax(q_values, dim=-1).item()
                
#                 reward = 0

#                 for i in range(self.step_repeat):
#                     reward_temp = 0

#                     next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

#                     reward += reward_temp

#                     if(done):
#                         break
                
#                 next_obs = self.process_observation(next_obs)

#                 self.memory.store_transition(obs, action, reward, next_obs, done)

#                 obs = next_obs

#                 episode_reward += reward
#                 episode_steps += 1
#                 total_steps += 1

#                 if self.memory.can_sample(batch_size):

#                     observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

#                     dones = dones.unsqueeze(1).float()

#                     # Current Q-Values from both models
#                     q_values = self.model(observations)
#                     actions = actions.unsqueeze(1).long()
#                     qsa_batch = q_values.gather(1, actions)

#                     next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)

#                     next_q_values = self.target_model(next_observations).gather(1, next_actions)

#                     target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

#                     loss = F.mse_loss(qsa_batch, target_b.detach())

#                     writer.add_scalar("Loss/model", loss.item(), total_steps)

#                     self.model.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()

#                     if episode_steps % 4 == 0:
#                         soft_update(self.target_model, self.model)
                    
            
#             self.model.save_the_model()

#             writer.add_scalar('Score', episode_reward, episode)
#             writer.add_scalar('Epsilon', epsilon, episode)

#             if epsilon > min_epsilon:
#                 epsilon *= epsilon_decay
            
#             episode_time = time.time() - episode_start_time

#             print(f"Completed episode {episode} with score {episode_reward}")
#             print(f"Episode Time: {episode_time:1f} seconds")
#             print(f"Episode Steps: {episode_steps}")

from buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import cv2


class Agent():

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):
        
        self.env = env
        self.step_repeat = step_repeat
        self.gamma = gamma

        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Loaded model on device {self.device}')

        self.memory = ReplayBuffer(
            max_size=500000,
            input_shape=obs.shape,
            device=self.device
        )

        self.model = Model(
            action_dim=env.action_space.n,
            hidden_dim=hidden_layer,
            observation_shape=obs.shape
        ).to(self.device)
        
        self.target_model = Model(
            action_dim=env.action_space.n,
            hidden_dim=hidden_layer,
            observation_shape=obs.shape
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        if not os.path.exists('models'):
            os.makedirs('models')


    def process_observation(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        return obs


    def save_checkpoint(self, episode, epsilon, total_steps, checkpoint_path='models/checkpoint.pth'):
        checkpoint = {
            'episode': episode,
            'epsilon': epsilon,
            'total_steps': total_steps,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'step_repeat': self.step_repeat,

            # Replay buffer
            'memory_mem_ctr': self.memory.mem_ctr,
            'memory_mem_size': self.memory.mem_size,
            'state_memory': self.memory.state_memory,
            'next_state_memory': self.memory.next_state_memory,
            'action_memory': self.memory.action_memory,
            'reward_memory': self.memory.reward_memory,
            'terminal_memory': self.memory.terminal_memory,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')


    def load_checkpoint(self, checkpoint_path='models/checkpoint.pth'):
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}')
            return 0, None, 0

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.memory.mem_ctr = checkpoint['memory_mem_ctr']
        self.memory.mem_size = checkpoint['memory_mem_size']
        self.memory.state_memory = checkpoint['state_memory']
        self.memory.next_state_memory = checkpoint['next_state_memory']
        self.memory.action_memory = checkpoint['action_memory']
        self.memory.reward_memory = checkpoint['reward_memory']
        self.memory.terminal_memory = checkpoint['terminal_memory']

        episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        total_steps = checkpoint['total_steps']

        print(f'Checkpoint loaded from {checkpoint_path}')
        print(f'Resuming from episode {episode + 1}, epsilon={epsilon}, total_steps={total_steps}')

        return episode + 1, epsilon, total_steps


    def test(self):
        self.model.load_the_model()

        done = False
        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        episode_reward = 0

        while not done:

            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                action = torch.argmax(q_values, dim=-1).item()
            
            reward = 0

            for i in range(self.step_repeat):
                reward_temp = 0
                next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

                reward += reward_temp

                frame = self.env.env.env.render()
                resized_frame = cv2.resize(frame, (500, 400))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("Pong AI", resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.05)

                if done:
                    break
            
            obs = self.process_observation(next_obs)
            episode_reward += reward

        print(f"Test episode reward: {episode_reward}")


    def train(
        self,
        episodes,
        max_episode_steps,
        summary_writer_suffix,
        batch_size,
        epsilon,
        epsilon_decay,
        min_epsilon,
        resume=False,
        checkpoint_path='models/checkpoint.pth',
        checkpoint_interval=10
    ):

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        total_steps = 0
        start_episode = 0

        if resume:
            loaded_episode, loaded_epsilon, loaded_total_steps = self.load_checkpoint(checkpoint_path)
            start_episode = loaded_episode
            total_steps = loaded_total_steps

            if loaded_epsilon is not None:
                epsilon = loaded_epsilon

        for episode in range(start_episode, episodes):

            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            obs = self.process_observation(obs)

            episode_steps = 0
            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:

                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                    action = torch.argmax(q_values, dim=-1).item()
                
                reward = 0

                for i in range(self.step_repeat):
                    reward_temp = 0
                    next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

                    reward += reward_temp

                    if done:
                        break
                
                next_obs = self.process_observation(next_obs)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):

                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

                    dones = dones.unsqueeze(1).float()

                    q_values = self.model(observations)
                    actions = actions.unsqueeze(1).long()
                    qsa_batch = q_values.gather(1, actions)

                    next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                    next_q_values = self.target_model(next_observations).gather(1, next_actions)

                    target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                    loss = F.mse_loss(qsa_batch, target_b.detach())

                    writer.add_scalar("Loss/model", loss.item(), total_steps)

                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)

            self.model.save_the_model()

            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            if epsilon > min_epsilon:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)

            episode_time = time.time() - episode_start_time

            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:.1f} seconds")
            print(f"Episode Steps: {episode_steps}")

            # Save checkpoint periodically
            if (episode + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    episode=episode,
                    epsilon=epsilon,
                    total_steps=total_steps,
                    checkpoint_path=checkpoint_path
                )

        # Save final checkpoint after training ends
        self.save_checkpoint(
            episode=episodes - 1,
            epsilon=epsilon,
            total_steps=total_steps,
            checkpoint_path=checkpoint_path
        )

        writer.close()