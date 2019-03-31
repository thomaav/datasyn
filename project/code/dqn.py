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


ENV_NAME = 'CartPole-v0'
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


class DQN(nn.Module):
    def __init__(self, observation_shape, output_shape):
        super().__init__()

        self.fc1 = nn.Linear(observation_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        self.classifier = nn.Linear(64, output_shape)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.classifier(x)


    def predict(self, x):
        return self(x).max(1)[1].view(1, 1)


class DQNAgent(object):
    def __init__(self, nstates, nactions):
        self.nstates = nstates
        self.nactions = nactions
        self.model = DQN(self.nstates, self.nactions)
        self.model.eval()

        self.memory_capacity = 2500
        self.memory = deque(maxlen=self.memory_capacity)

        self.gamma = 0.95
        self.eps = 1.0
        self.eps_end = 0.05
        self.eps_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def memorize(self, transition):
        self.memory.append(transition)


    def action(self, state):
        if np.random.rand() <= self.eps:
            return torch.tensor([[random.randrange(self.nactions)]], dtype=torch.long)

        with torch.no_grad():
            flattened_length = state.shape[1]*state.shape[2]
            return self.model.predict(state.view(-1, flattened_length))


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

        states = states.view(-1, states.shape[1]*states.shape[2])
        next_states = next_states.view(-1, next_states.shape[1]*next_states.shape[2])

        q = self.model(states).gather(1, actions)
        next_max_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (self.gamma*next_max_q)

        loss = F.mse_loss(q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.eps > self.eps_end:
            self.eps *= self.eps_decay


    def model():
        pass


    def save(self, fp):
        torch.save(self.model.state_dict(), fp)


    def load(self, fp):
        self.model.load_state_dict(torch.load(fp))
        model.eval()


def main():
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    # Settings.
    screen_dims = 28
    nb_actions = env.action_space.n

    # Gogo.
    state_renderer = CartPoleScreenPreprocessor(env)

    # Create a DQN agent.
    agent = DQNAgent(screen_dims**2, nb_actions)

    # Run the DQN algorithm.
    # for t in range(50000):
    while True:
        env.reset()
        previous_screen = state_renderer.render_current_state(dims=screen_dims)
        current_screen = state_renderer.render_current_state(dims=screen_dims)
        env_state = current_screen - previous_screen

        for i in itertools.count():
            action = agent.action(env_state)
            _, reward, done, _ = env.step(action.item())

            previous_screen = current_screen
            current_screen = state_renderer.render_current_state(dims=screen_dims)

            if done:
                reward = -10
            else:
                next_state = current_screen - previous_screen

            agent.memorize((env_state, action, torch.FloatTensor([reward]), next_state))
            agent.experience_replay()

            env_state = next_state

            if done:
                print('Done after {} iterations'.format(i))
                break


if __name__ == '__main__':
    main()
