import os
import zipfile
import pandas as pd
from tqdm import tqdm
from PIL import Image
import shutil
import sys
import subprocess
import glob

# Create directories for the dataset
os.makedirs('data/train/male', exist_ok=True)
os.makedirs('data/train/female', exist_ok=True)
os.makedirs('data/val/male', exist_ok=True)
os.makedirs('data/val/female', exist_ok=True)
os.makedirs('data/celeba_temp', exist_ok=True)

# Install required packages if not already installed
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package} package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install kaggle package if not already installed
install_package("kaggle")

print("Downloading CelebA dataset from Kaggle...")
print("Note: You need to have a Kaggle account and API key set up.")
print("If you haven't set up your Kaggle API key yet, please follow these steps:")
print("1. Create a Kaggle account at https://www.kaggle.com if you don't have one")
print("2. Go to your Kaggle account settings (https://www.kaggle.com/settings)")
print("3. Scroll down to the 'API' section and click 'Create New API Token'")
print("4. This will download a kaggle.json file")
print("5. Create a .kaggle directory in your home folder: mkdir -p ~/.kaggle")
print("6. Move the downloaded file to ~/.kaggle/kaggle.json")
print("7. Set the correct permissions: chmod 600 ~/.kaggle/kaggle.json")

# Check if kaggle.json exists
kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
if not os.path.exists(kaggle_path):
    print(f"Error: Kaggle API key not found at {kaggle_path}")
    print("Please follow the instructions above to set up your Kaggle API key.")
    sys.exit(1)

# Check if we already have the dataset
if not os.path.exists("data/original/list_attr_celeba.csv") and not os.path.exists("data/list_attr_celeba.csv"):
    # Download the CelebA dataset from Kaggle
    print("Downloading CelebA dataset (this may take some time)...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "jessicali9530/celeba-dataset", 
                       "-p", "data", "--unzip"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Make sure your Kaggle API key is set up correctly.")
        sys.exit(1)
else:
    print("Dataset already downloaded, skipping download step.")

# Find the attribute file
attr_file = None
for possible_path in [
    "data/list_attr_celeba.csv", 
    "data/original/list_attr_celeba.csv",
    "data/list_attr_celeba.txt", 
    "data/original/list_attr_celeba.txt"
]:
    if os.path.exists(possible_path):
        attr_file = possible_path
        break

if attr_file is None:
    print("Error: Attribute file not found.")
    print("The Kaggle download might be incomplete.")
    sys.exit(1)

# Find the image directory
img_dir = None
for possible_path in [
    "data/img_align_celeba/img_align_celeba",
]:
    if os.path.exists(possible_path) and os.path.isdir(possible_path):
        # Check if there are actually images in this directory
        image_files = glob.glob(os.path.join(possible_path, "*.jpg"))
        if image_files:
            img_dir = possible_path
            break

if img_dir is None:
    print("Error: Image directory not found or empty.")
    print("The Kaggle download might be incomplete.")
    sys.exit(1)

print(f"Found attribute file: {attr_file}")
print(f"Found image directory: {img_dir}")
print(f"Sample images: {', '.join(os.path.basename(f) for f in glob.glob(os.path.join(img_dir, '*.jpg'))[:5])}")

# Process the attribute file to create a dataframe
print("Processing attribute file...")
file_extension = os.path.splitext(attr_file)[1]

if file_extension == '.csv':
    # Handle CSV format
    df = pd.read_csv(attr_file)
    # Ensure the image ID column is correctly identified
    if 'image_id' not in df.columns:
        # Try to find the image ID column
        possible_id_columns = [col for col in df.columns if 'id' in col.lower() or 'file' in col.lower() or 'name' in col.lower() or 'image' in col.lower()]
        if possible_id_columns:
            # Rename the first matching column to 'image_id'
            df = df.rename(columns={possible_id_columns[0]: 'image_id'})
        else:
            # If no suitable column is found, assume the first column is the image ID
            df = df.rename(columns={df.columns[0]: 'image_id'})
    
    # Check if 'Male' column exists
    if 'Male' not in df.columns:
        print("Error: 'Male' attribute not found in the CSV file.")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)
else:
    # Handle TXT format
    with open(attr_file, 'r') as f:
        lines = f.readlines()
    
    # First line contains the number of images
    num_images = int(lines[0].strip())
    # Second line contains the attribute names
    attr_names = lines[1].strip().split()
    
    # Create a dataframe to store the data
    data = []
    for i in range(2, len(lines)):
        line = lines[i].strip().split()
        if len(line) > 0:
            img_name = line[0]
            # Convert attributes from -1/1 to 0/1 for easier processing
            attrs = [1 if int(val) > 0 else 0 for val in line[1:]]
            data.append([img_name] + attrs)
    
    df = pd.DataFrame(data, columns=['image_id'] + attr_names)

# Print dataset information
print(f"Dataset loaded with {len(df)} images and {len(df.columns)} attributes")
print(f"First few rows of the dataset:")
print(df.head())

# Split into training (80%) and validation (20%) sets
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Clean existing directories to avoid mixing old and new data
for dir_path in ["data/train/male", "data/train/female", "data/val/male", "data/val/female"]:
    for file in glob.glob(os.path.join(dir_path, "*")):
        os.remove(file)

# Process training images
print("Processing training images...")
train_count = {'male': 0, 'female': 0}
for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    img_name = row['image_id']
    # In CelebA, 'Male' attribute is 1 for male, 0 for female (after our conversion)
    gender = 'male' if row['Male'] == 1 else 'female'
    src_path = os.path.join(img_dir, img_name)
    dst_path = f"data/train/{gender}/{img_name}"
    
    if os.path.exists(src_path):
        # Resize image to a standard size
        img = Image.open(src_path)
        img = img.resize((128, 128))
        img.save(dst_path)
        train_count[gender] += 1
    else:
        print(f"Warning: Image {img_name} not found at {src_path}")

# Process validation images
print("Processing validation images...")
val_count = {'male': 0, 'female': 0}
for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
    img_name = row['image_id']
    gender = 'male' if row['Male'] == 1 else 'female'
    src_path = os.path.join(img_dir, img_name)
    dst_path = f"data/val/{gender}/{img_name}"
    
    if os.path.exists(src_path):
        # Resize image to a standard size
        img = Image.open(src_path)
        img = img.resize((128, 128))
        img.save(dst_path)
        val_count[gender] += 1
    else:
        print(f"Warning: Image {img_name} not found at {src_path}")

# Check if we have enough images
if sum(train_count.values()) == 0 or sum(val_count.values()) == 0:
    print("Error: No images were processed. Please check the image directory and attribute file.")
    sys.exit(1)

print(f"Dataset prepared.")
print(f"Training: {sum(train_count.values())} images ({train_count['male']} male, {train_count['female']} female)")
print(f"Validation: {sum(val_count.values())} images ({val_count['male']} male, {val_count['female']} female)")
print("Original dataset files have been preserved in the data directory.")
