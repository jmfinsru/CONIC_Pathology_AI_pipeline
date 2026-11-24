from PIL import Image
from pathlib import Path
import os

"""
Create black squares for masks when the image contains no artefacts, but the script requires a mask as input
"""
def generate_squares(output_dir: Path | str):
    # Store the images
    os.makedirs(output_dir, exist_ok=True)

    # Image size
    width, height = 66568, 66472

    # Create a single black image
    black_img = Image.new("RGB", (width, height), (0, 0, 0))

    # Generate x black images
    x = 1
    for i in range(x):
        filename = f"Func043_ST_HE_40x_BF_01_black_mask.png"
        black_img.save(os.path.join(output_dir, filename))

    print("Created black images in:", output_dir)

if __name__ == "__main__":
    output_dir = Path("/media/jenny/Expansion/HE/40x/masks/")
    generate_squares(output_dir)