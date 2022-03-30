This is the implementation of paper *Unauthorized Parking Detection using Deep Networks at Real Time*

Run `pip install -r requirements.txt' to install the packages. 
Run the following commands to prepare weights:
```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
Run `illegal_parking_video.py` to perform the experiments (data not included).

Please cite the paper if you use the codes:

```
@inproceedings{chen2019unauthorized,
  title={Unauthorized Parking Detection using Deep Networks at Real Time},
  author={Chen, Weiling and Yeo, Chai Kiat},
  booktitle={2019 IEEE International Conference on Smart Computing (SMARTCOMP)},
  pages={459--463},
  year={2019},
  organization={IEEE}
}
```

Yolo_v3 codes are from: https://github.com/qqwweee/keras-yolo3
