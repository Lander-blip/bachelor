import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import gymnasium as gym
from gymnasium.envs.registration import register
from minihack import MiniHackNavigation, LevelGenerator
from minihack.tiles.window import Window

# ------------------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------------------

# Cropped region (pixel coordinates)
TOP_LEFT = (575, 111)
BOTTOM_RIGHT = (703, 241)

# Class IDs:
# 0 : player
# 1 : stone
# 2 : exit
# 3 : entrance
# 4 : monster
# 5 : lava

# ------------------------------------------------------------------------------
# Custom MiniHack Environment
# ------------------------------------------------------------------------------

class MiniHackCustomEnv(MiniHackNavigation):
    """
    Custom MiniHack environment with a pre-mapped level.
    """
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)

        # Level description using a static map.
        des_map2 = (
            """MAZE: "mylevel", ' '\n"""
            """FLAGS:premapped\n"""
            """GEOMETRY:center,center\n"""
            """MAP\n"""
            """||||||||\n"""
            """|...L..|\n"""
            """|......|\n"""
            """|..LL..|\n"""
            """|......|\n"""
            """|...L..|\n"""
            """|..LLL.|\n"""
            """||||||||\n"""
            """ENDMAP\n"""
            """REGION:(0,0,7,7),lit,"ordinary"\n"""
            """BRANCH:(0,0,7,7),(0,0,0,0)\n"""
            """STAIR:(6, 6),down\n"""
            """MONSTER: ('a',"killer bee"), (2,2)\n"""
        )

        # Create an alternative level via LevelGenerator (currently unused in des_map2)
        map_str = (
            """|||||||||||||||\n"""
            """|...L.L...L...|\n"""
            """|.....L.L.L...|\n"""
            """|...L.L...L...|\n"""
            """|.....L.L.L...|\n"""
            """|...L.L...L...|\n"""
            """|.......L.L...|\n"""
            """|...L.........|\n"""
            """|.....L.L.....|\n"""
            """|...L.L...L...|\n"""
            """|.....L.L.L...|\n"""
            """|...L.L...L...|\n"""
            """|.....L.L.L...|\n"""
            """|...L.L...L...|\n"""
            """|||||||||||||||\n"""
        )
        lvl_gen = LevelGenerator(map=map_str, lit=True)
        lvl_gen.set_start_rect((0, 0), (21, 13))
        lvl_gen.add_goal_pos((21, 13))
        lvl_gen.add_monster(name="killer bee", symbol="a", place=(16, 1))
        lvl_gen.add_monster(name="wolf", symbol="d", place=(19, 1))
        lvl_gen.add_monster(name="killer bee", symbol="a", place=(13, 1))
        lvl_gen.add_monster(name="wolf", symbol="d", place=(16, 13))
        lvl_gen.add_monster(name="killer bee", symbol="a", place=(19, 13))
        lvl_gen.add_monster(name="wolf", symbol="d", place=(13, 13))

        # Initialize the environment using the des_map2 string.
        super().__init__(*args, des_file=des_map2, **kwargs)


# ------------------------------------------------------------------------------
# BoundingBox Class
# ------------------------------------------------------------------------------

class BoundingBox:
    """
    Represents a bounding box defined on an 8x8 grid.
    The grid is mapped onto a cropped image.
    """
    # For grid-to-pixel conversion
    ANSI_WIDTH = 8
    ANSI_HEIGHT = 8
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    SCALE_X = IMG_WIDTH / ANSI_WIDTH
    SCALE_Y = IMG_HEIGHT / ANSI_HEIGHT

    def __init__(self, x, y, width=1, height=1, class_id=0, color='red'):
        """
        Parameters:
            x, y : Top-left corner in grid coordinates.
            width, height : Dimensions in grid units.
            class_id : Object class (for YOLO).
            color : Color for drawing.
        """
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.class_id = class_id
        self.color = color
        self.rect = None

    def draw(self, window, offset=0):
        """Draw the bounding box on the Matplotlib window with an expanded offset."""
        # Calculate scaled position and dimensions
        # pos_screen = ((self.x + 0.5) * self.SCALE_X, (self.y) * self.SCALE_Y - 0.5)
        # screen_width = (self.w * self.SCALE_X) + (2 * offset)
        # screen_height = (self.h * self.SCALE_Y) + (2 * offset)

        pos_screen = ((self.x) * self.SCALE_X, (self.y) * self.SCALE_Y)
        screen_width = self.w * self.SCALE_X
        screen_height = self.h * self.SCALE_Y


        # Adjust position to keep the box centered with the new dimensions
        pos_screen_adjusted = (pos_screen[0] - offset, pos_screen[1] - offset)
        print(f"on screen: {pos_screen_adjusted}, ansi coord: {(self.x, self.y)}, id={self.class_id}")

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
        """
        Convert bounding box to YOLO format (normalized coordinates).
        """
        x_center = ((self.x + self.w) * self.SCALE_X + offset) / self.IMG_WIDTH
        y_center = ((self.y + self.h / 2) * self.SCALE_Y + offset) / self.IMG_HEIGHT
        width_norm = ((self.w * self.SCALE_X) + (2 * offset)) / self.IMG_WIDTH
        height_norm = ((self.h * self.SCALE_Y) + (2 * offset)) / self.IMG_HEIGHT
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"


# ------------------------------------------------------------------------------
# Main Function and Helper Functions
# ------------------------------------------------------------------------------

def main():
    # Register and create the custom environment.
    register(id="MiniHack-CustomEnv-v0", entry_point="__main__:MiniHackCustomEnv")
    observation_keys = ("pixel", "message")
    env = gym.make('MiniHack-CustomEnv-v0', observation_keys=observation_keys, render_mode="ansi")
    window = Window("MiniHack the Planet!")
    counter = 0
    snap_rate = 1  # Save image and label every frame
    total_images = 800
    train_images = 600

    def save_image(img, img_name, save_dir="images"):
        """Save an image to the specified directory."""
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, f"{img_name}.png")
        plt.imsave(image_path, img)
        print(f"Saved image: {image_path}")

    def save_labels(boxes, img_name, save_dir="labels"):
        """Save bounding box labels in YOLO format."""
        os.makedirs(save_dir, exist_ok=True)
        label_path = os.path.join(save_dir, f"{img_name}.txt")
        with open(label_path, "w") as f:
            for box in boxes:
                f.write(box.to_yolo() + "\n")
        print(f"Saved labels: {label_path}")

    def special_print(render_str):
        """
        Process the ANSI render string to extract bounding boxes.
        Returns a list of BoundingBox objects.
        """
        processed_str = render_str.replace(" ", "")[7:-7] + "\n"
        print("---------------------")
        print(repr(processed_str))
        print("---------------------")
        boxes = []
        row, col = 0, 0
        for ch in processed_str:
            if ch == '\n':
                row += 1
                col = 0
            else:
                if ch == "@":
                    boxes.append(BoundingBox(col, row, class_id=0))
                elif ch == "`":
                    boxes.append(BoundingBox(col, row, color="blue", class_id=1))
                elif ch == ">":
                    boxes.append(BoundingBox(col, row, color="green", class_id=2))
                elif ch == "<":
                    boxes.append(BoundingBox(col, row, color="green", class_id=3))
                elif ch in ("r", "d", "a"):
                    boxes.append(BoundingBox(col, row, color="white", class_id=4))
                elif ch == "}":
                    boxes.append(BoundingBox(col, row, color="violet", class_id=5))
                col += 1
        return boxes

    def redraw(obs):
        nonlocal counter
        print("Rendering frame:")
        render_str = env.render()
        print(f"Render string length: {len(render_str)}")
        window.clear_rectangles()
        boxes = special_print(render_str)

        # Crop the image based on TOP_LEFT and BOTTOM_RIGHT
        x1, y1 = TOP_LEFT
        x2, y2 = BOTTOM_RIGHT
        cropped_img = obs["pixel"][y1:y2, x1:x2]
        msg = obs["message"]
        msg = msg[:np.where(msg == 0)[0][0]].tobytes().decode("utf-8")

        # Draw each detected bounding box
        for box in boxes:
            box.draw(window)

        # Save images and labels based on counter.
        if counter % snap_rate == 0:
            if counter < train_images:
                img_dir = "images/train"
                label_dir = "labels/train"
            else:
                img_dir = "images/val"
                label_dir = "labels/val"
            save_labels(boxes, f"frame_{counter}", save_dir=label_dir)
            save_image(cropped_img, f"frame_{counter}", save_dir=img_dir)
            if counter // snap_rate > total_images:
                exit()

        window.show_obs(cropped_img, msg)
        time.sleep(2)
        counter += 1

    def reset():
        """Reset the environment and render the first frame."""
        obs, info = env.reset()
        redraw(obs)
        return obs

    def step(action):
        """Take an action and redraw the frame."""
        obs, reward, done, truncated, info = env.step(action)
        print(f"Observation pixel shape: {obs['pixel'].shape}")
        if done:
            print("Episode Completed!")
            obs = reset()
        else:
            redraw(obs)
        return done

    # Main event loop
    obs = reset()
    while True:
        action = env.action_space.sample()  # Random action
        reset()

    env.close()
    window.close()


if __name__ == "__main__":
    main()
