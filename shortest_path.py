import numpy as np

from time import sleep
from PIL import Image, ImageDraw


def find_shortest_path(img, x1, y1, x2, y2):
    sleep(1)
    print(type(x1), x2, y1, y2)
    coordinates = [(int(x1), int(y1)), (307, 199), (350, 250), (400, 320),
                   (450, 380), (512, 429), (int(x2), int(y2))]
    img_with_path = draw_shortest_path(img, coordinates)
    return img_with_path, "example_message"


def draw_shortest_path(img, coordinates):
    img_copy = Image.fromarray(img)
    draw = ImageDraw.Draw(img_copy)
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)
    return np.array(img_copy)
