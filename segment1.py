import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import sys


# img_path = glob.glob("flower_photos/Tulip/*.jpg") 
path1="flower_photos"
dirs=os.listdir(path1)
for files in dirs:
    file='Bluebell'
    img_path = glob.glob("flower_photos/"+file+"/*.jpg") 
    print(img_path)
    print("\n")
    length=len(file)+15
    print(length)
    for filename in img_path:
            path = filename[length:].split('.')[0]
            print(path)
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sp = img.shape
            print (sp)
            sz1 = sp[0]#height(rows) of image
            sz2 = sp[1]#width(colums) of image
            sz3 = sp[2]#the pixels value is made up of three primary colors

            mask=np.zeros(img.shape[:2],np.uint8)
            bgdModel=np.zeros((1,65),np.float64)
            fgdModel=np.zeros((1,65),np.float64)
            rect=(1,1,sp[1],sp[0])
            cv2.grabCut(img,mask,rect,bgdModel,fgdModel,20,cv2.GC_INIT_WITH_RECT)
            mask2=np.where((mask==2)|(mask==0),0,1)
            img=img*mask2[:,:,np.newaxis]
            plt.figure(figsize=(16,10))
            plt.subplot(121), plt.imshow(img)
            plt.title("grabcut")
            path_1 = "flower_photos/"+file+"/" + path + "_segment.jpg"
            print (path_1)
            fig = plt.gcf()

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            fig.savefig(path_1,format='jpg', bbox_inches='tight')
            plt.title("grabcut")
