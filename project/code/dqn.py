import numpy as np
import gym
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


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

        # Normalize the image to make it fit well with
        # PyTorch. flatten() would require a copy. Pretends that the
        # single channel after grayscaling is the batch size.
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
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        return self.classifier(x)


class DQNAgent(object):
    def __init__(self):
        pass


    def model():
        pass


def main():
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n

    # Gogo.
    state_renderer = CartPoleScreenPreprocessor(env)

    env.reset()
    dims = 28
    screen = state_renderer.render_current_state(dims=dims)
    plt.imshow(screen.cpu().view(dims, dims).numpy(), interpolation='none', cmap='gray')
    plt.show()

    while True:
        pass


if __name__ == '__main__':
    main()
