To run the paper experiments, download the [uWave](https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary) data set to the `data` folder.

For example, run the following from the command line:

```
wget -P data https://www.timeseriesclassification.com/Downloads/UWaveGestureLibrary{X,Y,Z}.zip
cd data
for f in *.zip; do unzip "$f" -d "${f%.zip}"; done
rm -r UWaveGestureLibrary*.zip
```
