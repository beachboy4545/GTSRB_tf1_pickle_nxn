import os
from PIL import Image

# Define paths
og_dataset_dir = './Images/'
new_dataset_dir = './GTSRB101imgs/'


# Define the desired image size
#image_size = (N, N)
image_size = (101, 101)


# Loop through each class directory in your dataset
for class_dir in os.listdir(og_dataset_dir):
    class_path = os.path.join(og_dataset_dir, class_dir)
    images = os.listdir(class_path)

    img_count = int(len(images))

    os.makedirs(os.path.join(new_dataset_dir,class_dir), exist_ok=True)
    
    # Resize and save the images to their respective directories
    for image in images:
        if image.endswith(('.ppm')):
            img = Image.open(os.path.join(class_path, image))
            img = img.resize(image_size, Image.ANTIALIAS)
            img.save(os.path.join(new_dataset_dir, class_dir, image))
