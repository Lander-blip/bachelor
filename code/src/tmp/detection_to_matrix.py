import gymnasium as gym
from minihack import MiniHackNavigation, LevelGenerator
from gymnasium.envs.registration import register
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

class PredictionBox:
    def __init__(self, box):
        self.x_min = box.xyxy[0][0]
        self.y_min = box.xyxy[0][1]
        self.x_max = box.xyxy[0][2]
        self.y_max = box.xyxy[0][3]
        self.prob = box.conf[0]
        self.cls = int(box.cls[0])

    def __str__(self):
        return f"Box[({self.x_min}, {self.y_min}), ({self.x_max}, {self.y_max}), prob={self.prob}, class={self.cls}]"



# Define a custom MiniHack environment.
class MiniHackCustomEnv(MiniHackNavigation):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)
        des_map = """MAZE: "mylevel", ' '
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
STAIR:(6, 6),down
MONSTER: ('a',"killer bee"), (2,2)
"""
        super().__init__(*args, des_file=des_map, **kwargs)

def predictionToBoxes(bs):
    boxes = []
    for box in bs:
        tmp = PredictionBox(box)
        boxes.append(tmp)
        print(tmp)
    return boxes


def map_prediction_boxes_to_grid(prediction_boxes, image_width=1264, image_height=336, grid_rows=8, grid_cols=8):
    """
    Maps a list of prediction boxes to an 8x8 grid.
    
    Parameters:
      prediction_boxes (list): A list of objects with attributes `xyxy`, `cls`, and `conf`.
      image_width (int): Width of the image in pixels.
      image_height (int): Height of the image in pixels.
      grid_rows (int): Number of rows in the grid (default 8).
      grid_cols (int): Number of columns in the grid (default 8).
      
    Returns:
      grid (list of lists): An 8x8 array where each cell contains a tuple (class, probability).
                             If no prediction falls in a cell, the cell is (-1, 1.0).
    """
    # Initialize an 8x8 grid with default value (-1, 1.0)
    grid = [[(-1, 1.0) for _ in range(grid_cols)] for _ in range(grid_rows)]
    
    # Compute cell dimensions.
    cell_width = image_width / grid_cols
    cell_height = image_height / grid_rows
    
    for box in prediction_boxes:
        # Extract coordinates, predicted class, and confidence.
        x_min, y_min, x_max, y_max = box.x_min, box.y_min, box.x_max, box.y_max
        pred_class = box.cls
        conf = box.prob
        
        # Compute the center of the bounding box.
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        
        # Determine which grid cell the center falls into.
        col = int(center_x // cell_width)
        row = int(center_y // cell_height)
        
        # Clamp indices to ensure they are within grid bounds.
        col = max(0, min(col, grid_cols - 1))
        row = max(0, min(row, grid_rows - 1))
        
        # Update the grid cell.
        # If the cell is still the default value or the new detection has a higher confidence, update it.
        if grid[row][col][0] == -1 or conf > grid[row][col][1]:
            grid[row][col] = (pred_class, conf)
    
    return grid

def displayGrid(grid):
    for row in grid:
        print(row)


def main():
    # Register the custom environment.
    register(
        id="MiniHack-CustomEnv-v0",
        entry_point="__main__:MiniHackCustomEnv",
    )
    # Create the environment.
    env = gym.make(
        'MiniHack-CustomEnv-v0',
        observation_keys=("pixel", "message"),
        render_mode="rgb_array",
    )
    
    # Load the YOLO model from best.pt using ultralytics' YOLO.
    model = YOLO("best.pt") 
    
    # Set up matplotlib for interactive display.
    plt.ion()  # turn on interactive mode
    fig, ax = plt.subplots()
    
    # Reset the environment to start.
    obs, info = env.reset()
    
    while True:
        # Take a random action.
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Get the image from the observation (assumed to be in RGB format).
        top_left=(577, 112)
        bottom_right=(704, 239)
        x1, y1 = top_left
        x2, y2 = bottom_right
        img = obs["pixel"][y1:y2, x1:x2]
        
        # Run YOLO inference on the image.
        results = model(img, save=True)
        boxes = predictionToBoxes(results[0].boxes.cpu().numpy())
        grid = map_prediction_boxes_to_grid(boxes)
        displayGrid(grid)
        # Render predictions on the image. results[0].plot() returns an annotated image.
        rendered_img = results[0].plot()  
        
        # Clear previous axes and display the new image.
        ax.clear()
        ax.imshow(rendered_img)
        ax.axis("off")
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Small delay so the display updates visibly.
        time.sleep(2)
        
        # Reset environment if episode is over.
        if done or truncated:
            obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main()
