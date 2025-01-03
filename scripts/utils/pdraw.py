
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def scale(factor, center_pos, coordinate):
    return (center_pos + factor*(coordinate - center_pos))

def create_image(coords, image_size_x = 500, image_size_y = 500, scaling_factor = 50):
    img = Image.new("RGB", (image_size_x, image_size_y), (255, 255, 255))
    drawing = ImageDraw.Draw(img)
    sum_x = 0
    sum_y = 0
    middle_x = image_size_x/2
    middle_y = image_size_y/2
    for (x,y) in coords:
        sum_x += x
        sum_y += y
        
    center_x = sum_x/len(coords)
    center_y = sum_y/len(coords)
    font = ImageFont.load_default()
    drawing.text((image_size_x-(image_size_x/4), image_size_y-(image_size_y/12)), "Room no: ABCD12345", (0, 0, 0, 128), font)
    drawing.text((image_size_x/10, image_size_y-(image_size_y/12)), "Generated by Layoutinator 9000", (0, 0, 0, 128), font)
    for i, corner in enumerate(coords):
        (x1, y1) = corner
        (x2, y2) = coords[(i+1)%len(coords)]
        x1 = scale(scaling_factor, center_x, x1) + middle_x
        x2 = scale(scaling_factor, center_x, x2) + middle_x
        y1 = scale(scaling_factor, center_y, y1) + middle_y
        y2 = scale(scaling_factor, center_y, y2) + middle_y
        drawing.line([(x1, y1), (x2, y2)], 128, 2)

    img.show()

#xy_coords = [(-2, 1), (2, 1), (2, -1), (-2, -1), (-3, 3)]
#create_image(xy_coords)

