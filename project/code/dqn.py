import itertools
import random
import numpy as np
import gym
import cv2
from collections import deque
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ScreenPreprocessor(object):
    @staticmethod
    def resize(dims, img):
        return T.Compose([T.ToPILImage(),
                          T.Resize(dims, interpolation=Image.CUBIC),
                          T.Grayscale(num_output_channels=1),
                          T.ToTensor()])(img)


    def __init__(self, env):
        self.env = env


    # @abstractmethod
    def scale_screen(self, screen):
        pass


    def render_state(self, state, dims):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = self.scale_screen(screen)

        # # Normalize the image to make it fit well with
        # # PyTorch. flatten() would require a copy. Pretends that the
        # # single channel after grayscaling is the batch size.
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return type(self).resize(dims, screen)


class CartPoleScreenPreprocessor(ScreenPreprocessor):
    def __init__(self, env):
        super().__init__(env)


    def scale_screen(self, screen):
        nchannels, height, width = screen.shape

        # Remove the parts above and below the cart.
        screen = screen[:, int(height*0.42):int(height*0.79), :]

        # Center the image on the cart.
        view_width = int(screen.shape[1]);

        env_to_img_scale = width / (2*self.env.x_threshold)
        img_cart_offset = self.env.state[0]*env_to_img_scale
        img_cart_position = int(img_cart_offset + width/2)

        if img_cart_position < view_width // 2:
            img_cart_slice = slice(view_width)
        elif img_cart_position > (width - view_width // 2):
            img_cart_slice = slice(-view_width, None)
        else:
            img_cart_slice = slice(img_cart_position - view_width // 2,
                                   img_cart_position + view_width // 2)

        return screen[:, :, img_cart_slice]


class SimpleCartPoleScreenPreprocessor(ScreenPreprocessor):
    def __init__(self, env):
        super().__init__(env)


    def render_state(self, state):
        state = torch.FloatTensor([state]).view(1, -1)
        return state.to(DEVICE)


class SpaceInvadersScreenPreprocessor(ScreenPreprocessor):
    def __init__(self, env):
        super().__init__(env)


    # http://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
    def render_current_state(self, dims=84):
        screen = self.env.render(mode='rgb_array')
        screen = cv2.cvtColor(cv2.resize(screen, (84, 110)), cv2.COLOR_BGR2GRAY)
        screen = screen[26:110,:]
        screen = np.reshape(screen, (84, 84, 1))

        screen = torch.from_numpy(screen).type(torch.FloatTensor)
        return screen.to(DEVICE)


    def render_state(self, state, dims):
        screen = state
        screen = cv2.cvtColor(cv2.resize(screen, (84, 110)), cv2.COLOR_BGR2GRAY)
        screen = screen[18:102,:]
        screen = np.reshape(screen, (84, 84, 1))

        screen = torch.from_numpy(screen).type(torch.FloatTensor)
        screen = screen.float() / 255

        return screen.to(DEVICE)


    def scale_screen(self, screen):
        pass


class SimpleDQN(nn.Module):
    def __init__(self, observation_shape, output_shape):
        super().__init__()

        self.fc1 = nn.Linear(observation_shape, 256)
        self.classifier = nn.Linear(256, output_shape)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.classifier(x)


    def predict(self, x):
        return self(x).max(1)[1].view(1, 1)


class FCDQN(nn.Module):
    def __init__(self, observation_shape, output_shape):
        super().__init__()

        self.fc1 = nn.Linear(observation_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, output_shape)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.classifier(x)


    def predict(self, x):
        return self(x).max(1)[1].view(1, 1)


class CNNDQN(nn.Module):
    def __init__(self, height, width, n_channels, output_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(64*7*7, 512)
        self.classifier = nn.Linear(512, output_shape)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.classifier(x)


    def predict(self, x):
        return self(x).max(1)[1].view(1, 1)


class DQNAgent(object):
    def __init__(self, screen_dims, env):
        # self.env = env
        # self.state_dims = screen_dims
        # self.state_size = screen_dims**2
        # self.nactions = self.env.action_space.n

        self.env = env
        self.state_dims = screen_dims
        self.state_size = screen_dims**2
        self.nactions = self.env.action_space.n

        self.state_renderer = SpaceInvadersScreenPreprocessor(self.env)
        # self.state_renderer = CartPoleScreenPreprocessor(self.env)
        # self.state_renderer = SimpleCartPoleScreenPreprocessor(self.env)

        self.memory_capacity = 70000
        self.memory = deque(maxlen=self.memory_capacity)
        self.initial_memory = 10000

        self.stack_size = 4
        self.stacked_frames = deque([np.zeros((self.state_dims, self.state_dims), dtype=np.int)
                                     for i in range(self.stack_size)], maxlen=self.stack_size)


        # Old.
        # self.model = FCDQN(self.state_size, self.nactions)
        # self.model = SimpleDQN(self.state_size, self.nactions)

        self.policy_net = CNNDQN(self.state_dims, self.state_dims,
                                 self.stack_size, self.nactions).to(DEVICE)
        self.target_net = CNNDQN(self.state_dims, self.state_dims,
                                 self.stack_size, self.nactions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_update = 1000

        self.gamma = 0.99
        self.eps = 1.0
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay_rate = 0.00001
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Metrics.
        self.rewards = []
        self.qs = []


    def view(self, batch_size):
        if type(self.policy_net) == FCDQN or type(self.policy_net) == SimpleDQN:
            return (batch_size, -1)
        elif type(self.policy_net) == CNNDQN:
            return (batch_size, self.stack_size, self.state_dims, self.state_dims)


    def visualize_state(self):
        current_screen = self.state_renderer.render_current_state(dims=self.state_dims)
        plt.imshow(current_screen.squeeze(), cmap='gray')
        plt.show()


    def _visualize_state(self, state):
        plt.imshow(state.squeeze(), cmap='gray')
        plt.show()


    def stack_frames(self, state, reset=False):
        if reset:
            self.stacked_frames = deque([np.zeros((self.state_dims, self.state_dims), dtype=np.int)
                                         for i in range(self.stack_size)], maxlen=self.stack_size)

            for i in range(self.stack_size):
                self.stacked_frames.append(state)
        else:
            self.stacked_frames.append(state)

        return torch.stack(list(self.stacked_frames)).squeeze().to(DEVICE)


    def memorize(self, transition):
        self.memory.append(transition)


    def action(self, state, use_eps=True):
        if (np.random.rand() <= self.eps) and use_eps:
            return torch.tensor([[random.randrange(self.nactions)]], device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            return self.policy_net.predict(state.view(*self.view(batch_size=1)))


    def experience_replay(self, decay_step=0):
        if len(self.memory) < self.initial_memory:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states).to(DEVICE)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        non_final_next_states = torch.cat([s for s in next_states if s is not None]).to(DEVICE)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                next_states)), device=DEVICE, dtype=torch.uint8)

        states_view_shape = self.view(self.batch_size)
        states = states.view(*states_view_shape)
        non_final_states_view_shape = self.view(non_final_next_states.shape[0] // self.stack_size)
        non_final_next_states = non_final_next_states.view(*non_final_states_view_shape)

        q = self.policy_net(states).gather(1, actions)
        next_max_q = torch.zeros(self.batch_size, device=DEVICE)
        next_max_q[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_q = rewards + (self.gamma*next_max_q)

        loss = F.smooth_l1_loss(q, expected_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.eps > self.eps_end:
            self.eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.eps_decay_rate * decay_step)


    def train(self, steps, viz=False):
        current_step = 0
        current_episode = 1
        decay_step = 0

        while current_step < steps:
            _first_state = self.env.reset()
            first_state = self.state_renderer.render_state(state=_first_state, dims=self.state_dims)

            total_reward = 0
            env_state = self.stack_frames(first_state, reset=True)

            for i in itertools.count():
                if i % 30 == 0:
                    print('Another 30 iterations over')

                action = self.action(env_state)
                _next_state, reward, done, _ = self.env.step(action.item())

                total_reward += reward
                decay_step += 1

                if viz:
                    self.env.render()

                # We already get the state above. Don't do this.
                next_state = self.state_renderer.render_state(state=_next_state, dims=self.state_dims)
                next_env_state = self.stack_frames(next_state, reset=False)

                if done:
                    next_env_state = None

                self.memorize((env_state, action, torch.FloatTensor([reward]).to(DEVICE), next_env_state))
                self.experience_replay(decay_step)

                env_state = next_env_state

                if (current_step + i) % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    print('Episode {} done after with {} rewards, {}/{}, eps: {}, memory: {}'.
                          format(current_episode, total_reward, current_step+i,
                                 steps, self.eps, len(self.memory)))
                    current_step += i
                    current_episode += 1
                    self.rewards.append(total_reward)

                    # Save weights.
                    if current_episode % 20 == 0:
                        self.save('nets/' + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))

                    break


    def test(self, episodes=5, visualize=True):
        for t in range(episodes):
            _first_state = self.env.reset()
            first_state = self.state_renderer.render_state(state=_first_state, dims=self.state_dims)

            env_state = self.stack_frames(first_state, reset=True)

            for i in itertools.count():
                action = self.action(env_state, use_eps=False)
                _next_state, reward, done, _ = self.env.step(action.item())

                self.env.render()

                next_state = self.state_renderer.render_state(state=_next_state, dims=self.state_dims)
                env_state = self.stack_frames(next_state, reset=False)

                if done:
                    print('Evaluation: done after {} steps'.format(i))
                    break


    def save(self, fp):
        torch.save(self.policy_net.state_dict(), fp)


    def load(self, fp):
        self.policy_net.load_state_dict(torch.load(fp, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())


def main():
    # ENV_NAME = 'CartPole-v1'
    # ENV_NAME = 'Breakout-v4'
    ENV_NAME = 'Pong-v0'

    env = gym.make(ENV_NAME)

    # Settings.
    screen_dims = 84

    # Run.
    agent = DQNAgent(screen_dims=screen_dims, env=env)
    # agent.load('nets/dqn-agent.h5')
    agent.train(steps=5000000, viz=True)
    agent.test()

    # agent.save('nets/dqn-agent.h5')

    # plt.ylabel('Episode reward')
    # plt.xlabel('Training episodes')
    # plt.plot(agent.rewards)
    # plt.show()

    # agent = DQNAgent(screen_dims=screen_dims, env=env)
    # agent.load('dqn-agent.h5')
    # agent.test()


if __name__ == '__main__':
    main()
