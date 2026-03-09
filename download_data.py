"""
Download and extract Food-11 dataset
"""
import os
import zipfile
import subprocess
from config import DATA_DIR


def download_food11():
    """Download Food-11 dataset from Kaggle"""
    print("Downloading Food-11 dataset...")
    
    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: Kaggle CLI not installed.")
        print("Please install it with: pip install kaggle")
        print("And set up your API key: https://www.kaggle.com/docs/api")
        return False
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download dataset
    zip_path = os.path.join(DATA_DIR, "food11.zip")
    
    try:
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "vermaavi/food11",
            "-p", DATA_DIR
        ], check=True)
        
        print("Download complete!")
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        print("Extraction complete!")
        
        # Remove zip file
        os.remove(zip_path)
        
        # List contents
        print("\nDataset structure:")
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path):
                num_files = len(os.listdir(item_path))
                print(f"  {item}/ ({num_files} items)")
            else:
                print(f"  {item}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        return False


def verify_dataset():
    """Verify dataset structure"""
    required_dirs = ['training', 'validation', 'evaluation']
    
    print("\nVerifying dataset...")
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = os.path.join(DATA_DIR, dir_name)
        if os.path.exists(dir_path):
            num_images = sum([len(files) for _, _, files in os.walk(dir_path)])
            print(f"  ✓ {dir_name}/ ({num_images} images)")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            all_good = False
    
    if all_good:
        print("\n✅ Dataset verified successfully!")
    else:
        print("\n❌ Dataset verification failed!")
    
    return all_good


if __name__ == "__main__":
    if not verify_dataset():
        download_food11()
        verify_dataset()
