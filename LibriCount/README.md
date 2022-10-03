To run the paper experiments, download the [LibriCount](https://zenodo.org/record/1216072) data set to the `data` folder. Then unzip the folder and run the feature extraction script:


```
wget -P data -O LibriCount.zip https://zenodo.org/record/1216072/files/LibriCount10-0dB.zip\?download\=1
cd data
unzip "$1" -d "${1%.zip}"
rm -r LibriCount.zip
python feature_extract.py
```

