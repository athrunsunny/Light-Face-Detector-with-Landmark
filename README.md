# Light-Face-Detector

Model size under 500KB

### Model size comparison
- Comparison of several open source lightweight face detection models:

Model|model file size（MB）
------|--------
Official Retinaface-Mobilenet-0.25 (Mxnet) | 1.68
Ultra-Light-Fast-Generic-Face-Detector-1MB(version-slim)| **1.04**
facelite(our)| **0.5** 

## Datasets prepare

### 1.Generate VOC format training data set and training process (for detect) 

Download the wideface official website dataset or download the training set I provided and extract it into the ./data folder:

   The clean widerface data pack after filtering out the 10px*10px small face: [Baidu cloud disk (extraction code: cbiu)](https://pan.baidu.com/s/1MR0ZOKHUP_ArILjbAn03sw) 、[Google Drive](https://drive.google.com/open?id=1OBY-Pk5hkcVBX1dRBOeLI4e4OCvqJRnH )

### 2.Download the wideface official website dataset (for landmark)  

Copy label.txt under widerface_landmark_gt folder and paste it into wider_face

```Shell
  data/
    widerface_landmark_gt/
      test/
      train/
        label.txt
      val/
        label.txt
    wider_face/
      WIDER_test/
      WIDER_train/
        images/
        label.txt
      WIDER_val/
        images/
        label.txt
``` 

## How to start

### install requirements
```Shell
pip install -r requirements.txt
```

## Detect model

### train
```Shell
python train.py --batch-size 16 --epochs 300 --weights '' --optimizer SGD
```

### detect
```Shell
python detect.py --weights face_lite.pth 
```

### export
```Shell
python export.py --weights face_lite.pth 
```

## Pretrained model(det)

Pretrained model: [Baidu cloud disk (extraction code: cz51)](https://pan.baidu.com/s/1L8Ut0QTTAPPPmV2qG6XhEg)


## Landmark model

### train
```Shell
python landmark/train.py --batch-size 16 --epochs 300 --weights '' --optimizer SGD
```

### detect
```Shell
python landmark/detect.py --weights xxx.pth 
```

### export
```Shell
python landmark/export.py --weights xxx.pth 
```

## Pretrained model(lmk)

Pretrained model: coming soon
