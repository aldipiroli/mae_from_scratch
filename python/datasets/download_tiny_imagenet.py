import os
import urllib.request
import zipfile

def download_tiny_imagenet(destination_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(destination_dir, "tiny-imagenet-200.zip")

    os.makedirs(destination_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Zip file already exists, skipping download.")

    extract_path = os.path.join(destination_dir, "tiny-imagenet-200")
    if not os.path.exists(extract_path):
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    download_tiny_imagenet()