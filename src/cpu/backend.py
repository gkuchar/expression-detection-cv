from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def draw_emoji(img, emoji_img, position):
    emoji_img = emoji_img.resize((emoji_img.width // 2, emoji_img.height // 2))
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pil_img.paste(emoji_img, position, emoji_img)
    result = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

# Pixel Normalization Function
def normalize(img: np.ndarray) -> np.ndarray:
    min_val = img.min()
    max_val = img.max()
    denom = max_val - min_val if max_val != min_val else 1.0
    normalized = (img - min_val) / denom        # [0, 1] float
    return (normalized * 255).astype(np.uint8)  # back to uint8 for saving