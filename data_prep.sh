wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
mkdir datasets
mv DocLayNet_core.zip datasets/
cd datasets/ && ./../unzip_file.py DocLayNet_core.zip # && rm DocLayNet_core.zip
cd ../
python convert_dataset.py