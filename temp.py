import numpy as np
import cv2


temp_dir = './data_demo/patient0005_4CH_ED_gt.png'

temp_mask = cv2.imread(temp_dir)

cv2.imshow('tt',temp_mask*255)
cv2.waitKey()