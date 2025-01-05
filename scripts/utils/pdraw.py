
from PIL import Image, ImageDraw, ImageFont

def scale(factor, center_pos, coordinate):
    return (center_pos + factor*(coordinate - center_pos))

def create_image(coords, image_size_x = 500, image_size_y = 500, scaling_factor = 50, file_path = ""):
    img = Image.new("RGB", (image_size_x, image_size_y), (255, 255, 255))
    drawing = ImageDraw.Draw(img)
    middle_x = image_size_x/2
    middle_y = image_size_y/2

    sum_x = 0
    sum_y = 0
    for corner in coords:
        sum_x += corner[0]
        sum_y += corner[1]
        
    center_x = sum_x/len(coords)
    center_y = sum_y/len(coords)
    font = ImageFont.load_default(size=34)
    drawing.text((image_size_x-(image_size_x/2), image_size_y-(image_size_y/12)), "Room: Zillow Indoor Dataset (5)", (0, 0, 0, 128), font)
    for i, corner in enumerate(coords):
        x1 = corner[0]
        y1 = corner[1]
        x2 = coords[(i+1)%len(coords)][0]
        y2 = coords[(i+1)%len(coords)][1]

        x1 = scale(scaling_factor, center_x, x1) + middle_x
        x2 = scale(scaling_factor, center_x, x2) + middle_x
        y1 = scale(scaling_factor, center_y, y1) + middle_y
        y2 = scale(scaling_factor, center_y, y2) + middle_y
        drawing.line([(x1, y1), (x2, y2)], fill="black", width=4)
    img.save(file_path)

