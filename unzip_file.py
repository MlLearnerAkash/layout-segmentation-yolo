#!/usr/bin/env python3
import argparse
import zipfile
import tarfile
import os

def extract_zip(zip_file, target_directory):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_directory)
    print(f"Extracted {zip_file} to {target_directory}")

def extract_tar(tar_file, target_directory):
    with tarfile.open(tar_file, 'r') as tar_ref:
        tar_ref.extractall(target_directory)
    print(f"Extracted {tar_file} to {target_directory}")

def main():
    parser = argparse.ArgumentParser(description="Unzip or untar files to the specified directory")
    parser.add_argument('--file_name', help="Path to the ZIP or TAR file to be extracted")
    parser.add_argument('--target_directory', help="Directory where the files will be extracted")

    args = parser.parse_args()

    file_name_directory = args.file_name
    target_directory = args.target_directory

    for file_name in os.listdir(file_name_directory):
        if file_name.endswith(".zip") or file_name.endswith(".tar") or  file_name.endswith('.tar.gz') or file_name.endswith('.tgz'):
            #file_names.append(file_name)
            print(f"Extracting: {file_name}")
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            if file_name.endswith('.zip'):
                try:
                    extract_zip(os.path.join(file_name_directory,file_name), target_directory)
                except Exception as e:
                    print(f"An error occurred while extracting ZIP file: {e}")
            elif file_name.endswith('.tar') or file_name.endswith('.tar.gz') or file_name.endswith('.tgz'):
                try:
                    extract_tar(os.path.join(file_name_directory,file_name), target_directory)
                except Exception as e:
                    print(f"An error occurred while extracting TAR file: {e}")
            else:
                print("Unsupported file format. Please provide a .zip or .tar file.")

if __name__ == "__main__":
    main()
