import numpy as np
from PIL import Image


def split_multi_view_image(image_array):
    # image_array: shape (H, W, 12)
    front = image_array[:, :, 0:3]
    left = image_array[:, :, 3:6]
    right = image_array[:, :, 6:9]
    back = image_array[:, :, 9:12]

    # Convert to PIL images
    front_img = Image.fromarray(front.astype(np.uint8))
    left_img = Image.fromarray(left.astype(np.uint8))
    right_img = Image.fromarray(right.astype(np.uint8))
    back_img = Image.fromarray(back.astype(np.uint8))

    return front_img, left_img, right_img, back_img


def combine_views(front, left, right, back):
    width, height = front.size
    combined = Image.new("RGB", (width * 4, height))
    combined.paste(front, (0, 0))
    combined.paste(left, (width, 0))
    combined.paste(right, (width * 2, 0))
    combined.paste(back, (width * 3, 0))
    return combined


def get_view_image(img_np, view='right'):
    front_img, left_img, right_img, back_img = split_multi_view_image(img_np)

    view_map = {
        'front': front_img,
        'left': left_img,
        'right': right_img,
        'back': back_img,
        'multi': [front_img, left_img, right_img, back_img],
        'side': [left_img, right_img],
        'combined': combine_views(front_img, left_img, right_img, back_img)
    }

    if view not in view_map:
        raise ValueError(
            f"Invalid view '{view}'. Choose from 'front', 'left', 'right', 'back', 'combined', 'multi', 'side'.")

    return view_map[view]
