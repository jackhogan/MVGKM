To run the paper experiments, download the [MVSA-Single](https://portland-my.sharepoint.com/:u%3A/g/personal/shiaizhu2-c_my_cityu_edu_hk/Ebcsf1kUpL9Do_u4UfNh7CgBC19i6ldyYbDZwr6lVbkGQQ) data set to the `data` folder. Then unzip the folder and run the feature extraction script:


```
cd data
unzip "$1" -d "${1%.zip}"
rm -r MVSA-Single.zip
python feature_extract.py
```
