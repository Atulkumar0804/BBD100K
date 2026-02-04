"""
Script to download the BDD100K dataset from Google Drive.
This script checks if the dataset is already present and downloads/extracts it if missing.

Usage:
    python data_analysis/download_dataset.py
"""

import os
import sys
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

# Constants
FILE_ID = "1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT"
DESTINATION_DIR = Path(__file__).resolve().parents[1] / "data"
ARCHIVE_NAME = "bdd100k_dataset.zip"  # Assuming zip, will verify
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    # Get file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Downloading to {destination}...")
    
    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_archive(archive_path, extract_to):
    print(f"Extracting {archive_path} to {extract_to}...")
    
    if str(archive_path).endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif str(archive_path).endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        print(f"Unknown archive format: {archive_path}")
        return

    print("Extraction complete.")

def main():
    # Ensure data directory exists
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    
    archive_path = DESTINATION_DIR / ARCHIVE_NAME
    
    # Check if critical data directories already exist to skip download
    # Adjust these paths based on what's expected inside the zip
    expected_dir = DESTINATION_DIR / "bdd100k"
    
    if expected_dir.exists() and any(expected_dir.iterdir()):
        print(f"Dataset appears to be present in {expected_dir}. Skipping download.")
        
        # Optional: Ask user if they want to force redownload
        # choice = input("Force re-download? (y/N): ")
        # if choice.lower() != 'y':
        #     return
        return

    print(f"Dataset missing. Starting download from Google Drive (ID: {FILE_ID})...")
    
    try:
        download_file_from_google_drive(FILE_ID, archive_path)
        
        if archive_path.exists():
            extract_archive(archive_path, DESTINATION_DIR)
            
            # Optional: Remove archive after extraction to save space
            # print(f"Removing archive {archive_path}...")
            # os.remove(archive_path)
            
            print("✅ Dataset setup complete.")
        else:
            print("❌ Download failed. File not found.")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Try installing gdown: pip install gdown")
        print(f"And run: gdown --id {FILE_ID} -O {archive_path}")

if __name__ == "__main__":
    main()
