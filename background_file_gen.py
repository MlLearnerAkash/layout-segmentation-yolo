import os
from PIL import Image
import numpy as np

def add_noise_to_image(image, mean=0, stddev=5):
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image_array.shape)
    
    # Add the noise to the image
    noisy_image = image_array + noise
    
    # Clip the values to be in the valid range [0, 255] and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    return Image.fromarray(noisy_image)

def create_blank_images_and_labels(image_dir, label_dir, num_images=500):
    # Ensure directories exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i in range(num_images):
        # Create a blank white image
        image = Image.new('RGB', (1025, 1025), color='white')
        image = add_noise_to_image(image)
        image_name = f"{i}b.png"
        image_path = os.path.join(image_dir, image_name)

        # Save the image
        image.save(image_path)

        # Create a corresponding blank text file
        label_name = f"{i}b.txt"
        label_path = os.path.join(label_dir, label_name)

        # Save the empty text file
        with open(label_path, 'w') as f:
            f.write("")

if __name__ == "__main__":
    image_directory = "/home/akash/ws/layout-segmentation-yolo/datasets/back_ground/images"
    label_directory = "/home/akash/ws/layout-segmentation-yolo/datasets/back_ground/labels"
    create_blank_images_and_labels(image_directory, label_directory)
