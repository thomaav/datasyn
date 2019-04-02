import itertools
import random
import numpy as np
import gym
from collections import deque
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim


# ENV_NAME = 'CartPole-v0'
ENV_NAME = 'CartPole-v1'
# ENV_NAME = 'MountainCar-v0'


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


    def render_current_state(self, dims=40):
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
    def __init__(self, height, width, output_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.classifier = nn.Linear(5, output_shape)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.classifier(x)


class DQNAgent(object):
    def __init__(self, screen_dims, env):
        self.env = env
        self.state_dims = screen_dims
        self.state_size = screen_dims**2
        self.nactions = self.env.action_space.n

        self.state_renderer = CartPoleScreenPreprocessor(self.env)
        # self.model = FCDQN(self.state_size, self.nactions)
        # self.model.eval()
        self.model = CNNDQN(self.state_dims, self.state_dims, self.nactions)
        self.model.eval()

        self.memory_capacity = 50000
        self.memory = deque(maxlen=self.memory_capacity)

        self.gamma = 0.95
        self.eps = 1.0
        self.eps_end = 0.1
        self.eps_decay = 0.999
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def visualize_state(self):
        current_screen = self.state_renderer.render_current_state(dims=self.state_dims)
        plt.imshow(current_screen.view(self.state_dims, self.state_dims), cmap='gray')
        plt.show()


    def visualize_state(self, state):
        plt.imshow(state.view(self.state_dims, self.state_dims), cmap='gray')
        plt.show()


    def memorize(self, transition):
        self.memory.append(transition)


    def action(self, state, use_eps=True):
        if np.random.rand() <= self.eps and use_eps:
            return torch.tensor([[random.randrange(self.nactions)]], dtype=torch.long)

        with torch.no_grad():
            return self.model.predict(state.view(1, -1))


    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Consider turning this into a list to avoid the quadratic
        # time random.sample?
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        states = states.view(self.batch_size, -1)
        next_states = next_states.view(self.batch_size, -1)

        q = self.model(states).gather(1, actions)
        next_max_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (self.gamma*next_max_q)

        loss = F.mse_loss(q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.eps > self.eps_end:
            self.eps *= self.eps_decay


    def train(self, steps):
        current_step = 0
        current_episode = 1

        while current_step < steps:
            self.env.reset()
            previous_screen = self.state_renderer.render_current_state(dims=self.state_dims)
            current_screen = self.state_renderer.render_current_state(dims=self.state_dims)
            env_state = current_screen - previous_screen

            for i in itertools.count():
                action = self.action(env_state)
                _, reward, done, _ = self.env.step(action.item())

                previous_screen = current_screen
                current_screen = self.state_renderer.render_current_state(dims=self.state_dims)

                if done:
                    reward = -10
                else:
                    next_state = current_screen - previous_screen

                self.memorize((env_state, action, torch.FloatTensor([reward]), next_state))
                self.experience_replay()

                env_state = next_state

                if done:
                    print('Episode {} done after {} iterations, {}/{}, eps: {}'.
                          format(current_episode, i, current_step+i, steps, self.eps))
                    current_step += i
                    current_episode += 1
                    break


    def eval(self, episodes=5, visualize=True):
        for t in range(episodes):
            self.env.reset()
            previous_screen = self.state_renderer.render_current_state(dims=self.state_dims)
            current_screen = self.state_renderer.render_current_state(dims=self.state_dims)
            env_state = current_screen - previous_screen

            for i in itertools.count():
                action = self.action(env_state, use_eps=False)
                _, reward, done, _ = self.env.step(action.item())

                previous_screen = current_screen
                current_screen = self.state_renderer.render_current_state(dims=self.state_dims)
                env_state = current_screen - previous_screen

                if done:
                    print('Evaluation: done after {} steps'.format(i))
                    break


    def save(self, fp):
        torch.save(self.model.state_dict(), fp)


    def load(self, fp):
        self.model.load_state_dict(torch.load(fp))
        self.model.eval()


def main():
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    env.reset()

    # Settings.
    screen_dims = 40

    # Run.
    agent = DQNAgent(screen_dims=screen_dims, env=env)
    # agent.load('nets/dqn-agent.h5')
    agent.train(steps=4000)
    agent.eval()

    # agent.save('nets/dqn-agent.h5')


if __name__ == '__main__':
    main()
