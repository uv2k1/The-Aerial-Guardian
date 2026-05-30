import os
import zipfile

def main():
    # Google Drive ID for VisDrone2019-MOT-val.zip
    file_id = "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_zip = "VisDrone2019-MOT-val.zip"
    extract_dir = "VisDrone2019-MOT-val"

    print("Checking if dataset zip already exists...")
    if not os.path.exists(output_zip):
        print(f"Downloading {output_zip} from Google Drive...")
        import importlib
        gdown_installed = False
        try:
            gdown = importlib.import_module("gdown")
            gdown_installed = True
        except ImportError:
            print("gdown package is not installed. Attempting to install it via pip...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
                importlib.invalidate_caches()
                gdown = importlib.import_module("gdown")
                gdown_installed = True
            except Exception as e:
                print(f"\nFailed to automatically install or run gdown: {e}")
                print("Please install gdown manually by running:")
                print("    pip install gdown")
                print("Then run this script again.")
                sys.exit(1)
        
        if gdown_installed:
            gdown.download(url, output_zip, quiet=False)
    else:
        print(f"{output_zip} already exists, skipping download.")

    print(f"Extracting {output_zip} to {extract_dir}...")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print("Dataset extracted successfully!")
    
    # List files to verify
    print("\nContents of extracted directory:")
    for item in os.listdir(extract_dir)[:10]:
        print(f" - {item}")

if __name__ == "__main__":
    main()
