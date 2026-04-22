# from agent import Agent
# import gymnasium as gym
# from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
# import ale_py


# episodes = 10000
# max_episode_steps = 10000
# hidden_layer = 128
# learning_rate = 0.0001
# step_repeat = 4
# gamma = 0.99
# batch_size = 64
#epsilon = 1
# min_epsilon = 0.1
# epsilon_decay = 0.995


# env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

# env = ResizeObservation(env, (64, 64))

# env = GrayscaleObservation(env, keep_dim=True)

# agent = Agent(env, hidden_layer=hidden_layer,
#               learning_rate=learning_rate, step_repeat=step_repeat,
#               gamma=gamma)

# summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}'

# agent.train(episodes=episodes,
#             max_episode_steps=max_episode_steps,
#             summary_writer_suffix=summary_writer_suffix,
#             batch_size=batch_size,
#             epsilon=epsilon,
#             epsilon_decay=epsilon_decay,
#             min_epsilon=min_epsilon)
from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py


episodes = 10000
max_episode_steps = 10000
hidden_layer = 128
learning_rate = 0.0001
step_repeat = 4
gamma = 0.99
batch_size = 64
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995

resume = True
checkpoint_path = 'models/checkpoint.pth'
checkpoint_interval = 10

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = ResizeObservation(env, (64, 64))
env = GrayscaleObservation(env, keep_dim=True)

agent = Agent(
    env,
    hidden_layer=hidden_layer,
    learning_rate=learning_rate,
    step_repeat=step_repeat,
    gamma=gamma
)

summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}'

try:
    agent.train(
        episodes=episodes,
        max_episode_steps=max_episode_steps,
        summary_writer_suffix=summary_writer_suffix,
        batch_size=batch_size,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        resume=resume,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval
    )
except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    env.close()