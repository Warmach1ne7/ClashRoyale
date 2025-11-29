import cv2
import numpy as np
from cr_element import crop_arena
# Load your image (replace 'input.png' with your actual image path)

img = cv2.imread('/home/ostikar/MyProjects/CS541/ClashRoyale/hf_subset/arena_31/ffd6fc01-fd2f-4e47-b787-62ef4cb30540/images/frame_000800.png')
if img is None:
    raise FileNotFoundError("Image not found. Please check the path.")
arena_crop = crop_arena(img)
# Show the cropped image
cv2.imshow("Arena Crop", arena_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the cropped image
cv2.imwrite('arena_crop.png', arena_crop)