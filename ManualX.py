import cv2
from os import listdir
import numpy as np
n=1
root='/home/kawsar/aA-WORKING/archive/PlantVillage/Generate/Cercospora leaf spot/'
imgs=listdir(root)

constX=6
rootX='/home/kawsar/aA-WORKING/archive/RegenaratedX/Brinjal/Cercospora leaf spot/'
for img in imgs:
    name=rootX+str(n)+'.jpg'
    print(img)
    image_rgb = cv2.imread(root+img)
    #image_bgr=cv2.resize(image_bgr,(250,250))
    #image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    rectangle = (18, 18, 220, 220)
    mask = np.zeros(image_rgb.shape[:2], np.uint8,order='C')

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, constX, cv2.GC_INIT_WITH_RECT)

    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image_rgd_nobg = image_rgb * mask_2[:, :, np.newaxis]
    # h ,w ,_ = image_rgd_nobg.shape
    # for i in range(h):
    #     for j in range(w):
    #         if image_rgd_nobg[i,j].sum() == 0:
    #             image_rgd_nobg[i,j]=[255,255,255]
    cv2.imwrite(name,image_rgd_nobg)

    n=n+1
    
