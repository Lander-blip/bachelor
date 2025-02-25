#!/usr/bin/env python3
import argparse
import numpy as np
import gymnasium as gym
from minihack import MiniHackNavigation, LevelGenerator  # noqa: F401
from minihack.tiles.window import Window 
from minihack.envs import register
from minihack import MiniHackNavigation, LevelGenerator
from nle.nethack import Command

#level generator at https://minihack-editor.github.io/
class MiniHackCustomEnv(MiniHackNavigation):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)

        map = """
        |||||||||||||||
        |...L.L...L...|
        |.....L.L.L...|
        |...L.L...L...|
        |.....L.L.L...|
        |...L.L...L...|
        |.......L.L...|
        |...L.........|
        |.....L.L.....|
        |...L.L...L...|
        |.....L.L.L...|
        |...L.L...L...|
        |.....L.L.L...|
        |...L.L...L...|
        |||||||||||||||
        """

        lvl_gen = LevelGenerator(map=map, lit=True)
        lvl_gen.set_start_rect((0, 0), (9, 1))

        lvl_gen.add_goal_pos((21, 13))

        super().__init__(*args, des_file=lvl_gen.get_des(), **kwargs)


def main():
    register(
    id="MiniHack-CustomEnv-v0",
    entry_point="__main__:MiniHackCustomEnv",
    )


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gymnasium environment to load",
        default="MiniHack-CustomEnv-v0",
    )
    args = parser.parse_args()

    observation_keys = ("pixel", "message")
    env = gym.make(args.env, observation_keys=observation_keys)

    def reset():
        obs, info = env.reset()
        redraw(obs)

    def step(action):
        obs, reward, done, truncated, info = env.step(action)
        if done:
            print("Episode Completed!")
            reset()
        else:
            redraw(obs)

    def key_handler(event):
        key_map = {
            "left": "h",
            "right": "l",
            "up": "k",
            "down": "j",
            "escape": 27
        }

        key = event.key
        if key == "escape":
            window.close()
            return

        if key == "backspace":
            reset()
            return

        if key == "space":
            step(env.action_space.sample())
            return

        action_char = key_map.get(key, key)

        if len(action_char) > 1:
            print(f"Key {key} is not mapped to a single action.")
            return

        try:
            action = ord(action_char)  # Convert character to integer
            if hasattr(env, 'actions'):
                #action_index = env.actions.index(action)
                step(env.action_space.sample())
            else:
                print(f"The environment does not have 'actions' attribute. Available attributes: {dir(env)}")
        except (ValueError, TypeError) as e:
            print(f"Error processing action {action_char} ({key}): {str(e)}")


    window = Window("MiniHack the Planet - " + args.env)
    window.reg_key_handler(key_handler)

    def redraw(obs):
        img = obs["pixel"]
        msg = obs["message"]
        msg = msg[:np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
        window.show_obs(img, msg)

    reset()
    window.show(block=True)

if __name__ == "__main__":
    main()
