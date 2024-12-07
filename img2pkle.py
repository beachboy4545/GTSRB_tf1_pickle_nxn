import os
import random
import pickle
from PIL import Image
import numpy as np

# Define paths
dataset_dir = './Images/'
train_dir = './mypickle/test/'
val_dir = './mypickle/valid/'
test_dir = './mypickle/train/'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define your split ratios
train_ratio = 0.67
val_ratio = 0.09
test_ratio = 0.24

# Define the desired image size
image_size = (101, 101)

# Loop through each class directory in your dataset
for class_dir in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_dir)
    images = os.listdir(class_path)
    random.shuffle(images)
    
    # Split the images for the class into train, validation, and test sets
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)
    test_count = int(len(images) * test_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count+val_count]
    test_images = images[train_count+val_count:]
    
    # Create subdirectories for the class in each split
    os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_dir), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)
    
    # Resize and save the images to their respective directories
    for image in train_images:
        if image.endswith(('.ppm')):
            img = Image.open(os.path.join(class_path, image))
            img = img.resize(image_size, Image.ANTIALIAS)
            img.save(os.path.join(train_dir, class_dir, image))
    for image in val_images:
        if image.endswith(('.ppm')):
            img = Image.open(os.path.join(class_path, image))
            img = img.resize(image_size, Image.ANTIALIAS)
            img.save(os.path.join(val_dir, class_dir, image))
    for image in test_images:
        if image.endswith(('.ppm')):
            img = Image.open(os.path.join(class_path, image))
            img = img.resize(image_size, Image.ANTIALIAS)
            img.save(os.path.join(test_dir, class_dir, image))

# Now, let's pickle the datasets
train_data = {"features": [], "labels": []}
val_data = {"features": [], "labels": []}
test_data = {"features": [], "labels": []}

# Populate the data dictionaries with NumPy arrays for image data and labels
for class_label in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_label)
    for image in os.listdir(class_path):
        img = Image.open(os.path.join(class_path, image))
        img = img.resize(image_size, Image.ANTIALIAS)
        img_array = np.array(img)
        label_array = np.array(class_label)
        print('img shape: ',np.shape(img_array))
        train_data["features"].append(img_array)
        train_data["labels"].append(label_array)

for class_label in os.listdir(val_dir):
    class_path = os.path.join(val_dir, class_label)
    for image in os.listdir(class_path):
        img = Image.open(os.path.join(class_path, image))
        img = img.resize(image_size, Image.ANTIALIAS)
        img_array = np.array(img)
        label_array = np.array(class_label)
        val_data["features"].append(img_array)
        val_data["labels"].append(label_array)

for class_label in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_label)
    for image in os.listdir(class_path):
        img = Image.open(os.path.join(class_path, image))
        img = img.resize(image_size, Image.ANTIALIAS)
        img_array = np.array(img)
        label_array = np.array(class_label)
        test_data["features"].append(img_array)
        test_data["labels"].append(label_array)

# Convert the entire 'features' list to a NumPy array before pickling
print('train_ data shape: ', np.shape(train_data["features"]))
train_data["features"] = np.array(train_data["features"])
val_data["features"] = np.array(val_data["features"])
test_data["features"] = np.array(test_data["features"])

# Serialize and save the datasets using pickle

with open('101train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('101val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)
with open('101test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

