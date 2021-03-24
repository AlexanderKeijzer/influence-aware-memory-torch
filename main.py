from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import random
import gym
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy

from environments.warehouse.warehouse import Warehouse


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        fnn_hidden_layer: int = 512,
        fnn_last_layer: int = 256,
        rnn_last_layer: int = 128,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = fnn_last_layer + rnn_last_layer
        self.latent_dim_vf = self.latent_dim_pi

        # Make a nicer way to do this
        self.d_set = [0, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
           63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

        self.fnn = nn.Sequential (
            nn.Linear(feature_dim, fnn_hidden_layer), nn.ReLU(),
            nn.Linear(fnn_hidden_layer, fnn_last_layer)
        )

        self.rnn = nn.LSTMCell(len(self.d_set), rnn_last_layer)

        self.out_relu = nn.ReLU()

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        #print(features.shape)

        y_fnn = self.fnn(features)
        y_rnn, _ = self.rnn(features[:, self.d_set])
        y = self.out_relu(th.cat((y_fnn, y_rnn), dim=1))

        return y, y


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

def main():
    seed = 0
    env = Warehouse(seed, {"num_frames": 1})
    env.reset()

    #model = PPO("MlpPolicy", env, verbose=1)
    model = PPO(CustomActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    """
    for i in range(100):
        obs, reward, done, _ = env.step(random.randint(0, 3))
        env.render(delay=0.001)
    """
    env.close()


if __name__ == "__main__":
    main()