# IT_THESIS_FINAL
Deepfake Image Detection

Dataset Link : 
  Initial Dataset: https://www.kaggle.com/datasets/iamshahzaibkhan/deepfake-database
  Celeb-df: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2/data
  FaceForensics: https://www.kaggle.com/datasets/xdxd003/ff-c23

Steps to Extract Frames from video:
1. Import libraries
2. define video dataset directory
3. defien output directory
4. set frame interval
5. set image size

Step to Execute Model:
1. Import libraries
2. Set configuration (model name, data directory, epoch, batch size, seed, fine_tune base, warmup epochs, learning rate, fine tune learning rate and output directory)
3. apply augmentation if any
4. set the model name in saving model section

Python IDE (Preferred Pycharm or Google Colab Pro)
Libraries:
  For Frames: cv2, os, shutil, sklearn
  For Model:os, math, json, itertools, numpy, tensorflow, sklearn
