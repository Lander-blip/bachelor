import gymnasium as gym
from minihack import MiniHackNavigation, LevelGenerator
from gymnasium.envs.registration import register
import numpy as np
from minihack.tiles.window import Window
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

np.set_printoptions(threshold=np.inf, linewidth=200)

# class id to object
# 0 : player
# 1 : stone
# 2 : exit
# 3 : entrance
# 4 : monster
# 5 : lava

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

        des_map2 = """MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
||||||||
|...L..|
|......|
|..LL..|
|......|
|...L..|
|..LLL.|
||||||||
ENDMAP
REGION:(0,0,7,7),lit,"ordinary"
BRANCH:(0,0,7,7),(0,0,0,0)
STAIR:(6, 6),down
MONSTER: ('a',"killer bee"), (2,2)
"""

        # Create a level using the LevelGenerator (this is currently unused in the des_file)
        lvl_gen = LevelGenerator(map=map, lit=True)
        lvl_gen.set_start_rect((0, 0), (21, 13))  # (9,1) for top left corner

        lvl_gen.add_goal_pos((21, 13))
        lvl_gen.add_monster(name="killer bee", symbol="a", place=(16,1))
        lvl_gen.add_monster(name="wolf", symbol="d", place=(19,1))
        lvl_gen.add_monster(name="killer bee", symbol="a", place=(13,1))
        lvl_gen.add_monster(name="wolf", symbol="d", place=(16,13))
        lvl_gen.add_monster(name="killer bee", symbol="a", place=(19,13))
        lvl_gen.add_monster(name="wolf", symbol="d", place=(13,13))

        # For this example we use des_map2.
        super().__init__(*args, des_file=des_map2, **kwargs)

class BoundingBox:
    ANSI_WIDTH = 80
    ANSI_HEIGHT = 21
    IMG_WIDTH = 1264
    IMG_HEIGHT = 336

    SCALE_X = IMG_WIDTH / ANSI_WIDTH  # 15.8
    SCALE_Y = IMG_HEIGHT / ANSI_HEIGHT  # 16.0

    def __init__(self, x, y, width=1, height=1, class_id=0, color='red'):
        """
        Parameters:
        - x, y: Top-left corner of the bounding box (ANSI coordinates).
        - width, height: Width and height in ANSI units.
        - class_id: Object class for YOLO format.
        - color: Color for visualization.
        """
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.color = color
        self.rect = None  # Store rectangle reference
        self.class_id = class_id

    def draw(self, window, offset=0):
        """Draw the bounding box on the Matplotlib window with an expanded offset."""
        # Calculate scaled position and dimensions
        pos_screen = ((self.x + 0.5) * self.SCALE_X, (self.y) * self.SCALE_Y - 0.5)
        screen_width = (self.w * self.SCALE_X) + (2 * offset)
        screen_height = (self.h * self.SCALE_Y) + (2 * offset)

        # Adjust position to keep the box centered with the new dimensions
        pos_screen_adjusted = (pos_screen[0] - offset, pos_screen[1] - offset)

        # Draw rectangle
        self.rect = patches.Rectangle(
            pos_screen_adjusted, screen_width, screen_height, 
            linewidth=2, edgecolor=self.color, facecolor='none'
        )
        window.ax.add_patch(self.rect)
        window.fig.canvas.draw()

        # Store rectangle in window for removal later
        window.rectangles.append(self.rect)

    def to_yolo(self, offset=0):
        """Convert bounding box to YOLO format (normalized values) with expanded offset."""
        x_center = ((self.x + self.w) * self.SCALE_X + offset) / self.IMG_WIDTH
        y_center = ((self.y + self.h / 2) * self.SCALE_Y + offset) / self.IMG_HEIGHT
        width = ((self.w * self.SCALE_X) + (2 * offset)) / self.IMG_WIDTH
        height = ((self.h * self.SCALE_Y) + (2 * offset)) / self.IMG_HEIGHT
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def main():
    register(
        id="MiniHack-CustomEnv-v0",
        entry_point="__main__:MiniHackCustomEnv",
    )

    # Global parameters: Total images and training images count.
    total_images = 800
    train_images = 600  # The rest (total_images - train_images) will be for validation.

    # Initialize the environment with observation keys.
    observation_keys = ("pixel", "message")
    env = gym.make(
        'MiniHack-CustomEnv-v0',
        observation_keys=observation_keys,
        render_mode="ansi",
    )
    crop_bounds = (30, 450, 30, 590)
    # window is 1264x336; in ansi render it's 80x21
    window = Window("MiniHack the Planet!")
    counter = 0
    snap_rate = 1  # Save image and label every snap_rate frames

    def save_image(img, img_name, save_dir="images"):
        """Save the Matplotlib figure as an image for YOLO dataset."""
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, f"{img_name}.png")
        plt.imsave(image_path, img)
        print(f"Saved image: {image_path}")

    def save_labels(boxes, img_name, save_dir="labels"):
        """Save bounding boxes in YOLO format."""
        os.makedirs(save_dir, exist_ok=True)
        label_path = os.path.join(save_dir, f"{img_name}.txt")
        with open(label_path, "w") as f:
            for box in boxes:
                f.write(box.to_yolo() + "\n")
        print(f"Saved labels: {label_path}")

    def special_print(string):
        boxes = []
        for i in range(len(string)):
            if string[i] == " ":
                print("", end="")
            else:
                print(string[i], end="")
            if string[i] == "@":  # player
                boxes.append(BoundingBox(i % 80, i // 80, class_id=0))
            elif string[i] == "`":  # stone
                boxes.append(BoundingBox(i % 80, i // 80, color="blue", class_id=1))
            elif string[i] == ">":  # exit
                boxes.append(BoundingBox(i % 80, i // 80, color="green", class_id=2))
            elif string[i] == "<":  # entrance
                boxes.append(BoundingBox(i % 80, i // 80, color="green", class_id=3))
            elif string[i] in ("r", "d", "a"):  # monster
                boxes.append(BoundingBox(i % 80, i // 80, color="white", class_id=4))
            elif string[i] == "}":  # lava
                boxes.append(BoundingBox(i % 80, i // 80, color="violet", class_id=5))
            elif string[i] == '\n':
                print("0\n", end="")
            else:
                print(string[i], end="")
        return boxes

    def redraw(obs):
        nonlocal counter
        print("render:")
        render = env.render()
        print(f"length string: {len(render)}")
        window.clear_rectangles()
        boxes = special_print(render)

        img = obs["pixel"]
        msg = obs["message"]
        msg = msg[:np.where(msg == 0)[0][0]].tobytes().decode("utf-8")

        for box in boxes:
            box.draw(window)

        # Save images and labels according to the training/validation threshold.
        if counter % snap_rate == 0:
            if counter < train_images:
                img_dir = "images/train"
                label_dir = "labels/train"
            else:
                img_dir = "images/val"
                label_dir = "labels/val"
            save_labels(boxes, f"frame_{counter}", save_dir=label_dir)
            save_image(img, f"frame_{counter}", save_dir=img_dir)
            if counter // snap_rate > total_images:
                exit()

        window.show_obs(img, msg)
        counter += 1

    def reset():
        """Reset the environment and update the display."""
        obs, info = env.reset()
        redraw(obs)
        return obs

    def step(action):
        """Perform an action in the environment and handle the result."""
        obs, reward, done, truncated, info = env.step(action)
        done = False
        print(obs["pixel"].shape)
        if done:
            print("Episode Completed!")
            obs = reset()
        else:
            redraw(obs)
        return done

    # Start the first episode
    obs = reset()

    # Main event loop
    while True:
        action = env.action_space.sample()  # Random action
        done = False
        reset()
        if done:
            break

    # Close the environment and clean up
    env.close()
    window.close()

if __name__ == "__main__":
    main()
