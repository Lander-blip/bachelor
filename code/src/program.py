#!/usr/bin/env python3
# This file is adapted from
# github.com/maximecb/gym-minigrid/blob/master/manual_control.py

import numpy as np
import gymnasium as gym
from minihack import MiniHackNavigation
from minihack.tiles.window import Window
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

np.set_printoptions(threshold=np.inf, linewidth=200)

# class id to object
# 0 : player
# 1 : stone 

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

    def draw(self, window, offset=1):
        """Draw the bounding box on the Matplotlib window with an expanded offset."""
        # Calculate scaled position and dimensions
        pos_screen = ((self.x + 0.5) * self.SCALE_X, (self.y) * self.SCALE_Y - 0.5)
        screen_width = (self.w * self.SCALE_X) + (2 * offset)  # Expand width with scaled offset
        screen_height = (self.h * self.SCALE_Y) + (2 * offset) # Expand height with scaled offset

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

    def to_yolo(self, offset=1):
        """Convert bounding box to YOLO format (normalized values) with expanded offset."""
        # Calculate center and size with offset added during scaling
        x_center = ((self.x + self.w / 2) * self.SCALE_X + offset) / self.IMG_WIDTH
        y_center = ((self.y + self.h / 2) * self.SCALE_Y + offset) / self.IMG_HEIGHT
        width = ((self.w * self.SCALE_X) + (2 * offset)) / self.IMG_WIDTH
        height = ((self.h * self.SCALE_Y) + (2 * offset)) / self.IMG_HEIGHT

        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"




def main():
    des_file = """
        
        MAZE: "mylevel", ' '
        FLAGS:premapped
        GEOMETRY:center,center

        MAP
        |||||||||||||
        |..L.L...L..|
        |....L.L.L..|
        |..L.L...L..|
        |......L.L..|
        |..L.L......|
        |....L.L....|
        |..L.L...L..|
        |....L.L.L..|
        |..L.L...L..|
        |....L.L.L..|
        |||||||||||||
        ENDMAP

        STAIR:(11, 5),down
        BRANCH: (1,7,1,7),(2,8,2,8)

        """


    # Initialize the environment with observation keys
    observation_keys = ("pixel", "message")
    env = env = gym.make(
    "MiniHack-Skill-Custom-v0",
    des_file=des_file,
    observation_keys=observation_keys,
    render_mode="ansi",
)
    crop_bounds = (30, 450, 30, 590)
    #window is 1264x336
    #in ansi render its 80x21
    window = Window("MiniHack the Planet!")
    counter = 0


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
        #string = string[(7*79):]
        boxes = []
        for i in range(len(string)):
            if string[i] == " ":
                print("", end="")
            else:
                print(string[i], end="")

            if string[i] == "@": #player
                boxes.append(BoundingBox(i % 80, i // 80, class_id=0))
            elif string[i] == "`": #stone
                boxes.append(BoundingBox(i % 80, i // 80, color="blue", class_id=1))
            elif string[i] == ">": #exit
                boxes.append(BoundingBox(i % 80, i // 80, color="green", class_id=2))
            elif string[i] == "<": #entrance
                boxes.append(BoundingBox(i % 80, i // 80, color="green", class_id=2))
            elif string[i] == "r": #entrance
                boxes.append(BoundingBox(i % 80, i // 80, color="white", class_id=2))
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

        # Get new bounding boxes
        boxes = special_print(render)

        img = obs["pixel"]
        msg = obs["message"]
        msg = msg[:np.where(msg == 0)[0][0]].tobytes().decode("utf-8")

        # upscale_factor = 2  # Adjust this as needed
        # img = cv2.resize(img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_NEAREST)

        # Draw bounding boxes
        for box in boxes:
            box.draw(window)

        # Save YOLO bounding box labels
        save_labels(boxes, f"frame_{counter}")

        # Save image
        save_image(img, f"frame_{counter}")

        window.show_obs(img, msg)
        #time.sleep(1)

        counter += 1



    def reset():
        """Reset the environment and update the display."""
        obs, info = env.reset()
        redraw(obs)
        return obs

    def step(action):
        """Perform an action in the environment and handle the result."""
        obs, reward, done, truncated, info = env.step(action)
        # print("-----------------------------------------------------------")
        # print(obs["pixel"])
        print(obs["pixel"].shape)
        # print("-----------------------------------------------------------")
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
        done = step(action)
        if done:
            break

    # Close the environment and clean up
    env.close()
    window.close()

if __name__ == "__main__":
    main()

