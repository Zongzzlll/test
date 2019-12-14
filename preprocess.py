import cv2
import numpy as np
import shutil
import os
import glob

def setDir(a):
    if os.path.exists(a):  # 若文件夹已存在
        shutil.rmtree(a)  # 先强制删除文件夹
        os.mkdir(a)  # 再重新建同名文件夹

    else:  # 若文件夹不存在
        os.mkdir(a)  # 新建文件夹


setDir('images_analysis_all')
setDir('images_analysis_clear')
setDir('threshold_analysis')

img_path = glob.glob("test/*.jpg")  # 路径中的所有jpg格式的图片
i = 1


# Laplacian算子是用来衡量图片的二阶导，能够强调图片中密度快速变化的区域，也就是边界，故常用于边界检测
# 在正常图片中边界比较清晰因此方差会比较大；而在模糊图片中包含的边界信息很少，所以方差会较小
# 用图片的1个通道用以下3x3的核进行卷积，然后计算输出的方差，如果方差小于一定值则图片视为模糊

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def motion_blur(image, degree=200, angle=60):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


for a_path in img_path:
    path = a_path[5:].split('.')[0]

    image = cv2.imread(a_path)
    path_1 = "images_analysis_all/" + path + ".jpg"
    cv2.imwrite(path_1, image)

    # 将原图写入到images_analysis文件夹中

    path_2 = "images_analysis_all/" + path + "_MotionBlur" + ".jpg"
    img_ = motion_blur(image)
    cv2.imwrite(path_2, img_)

    # 将原图经过运动模糊后写入到images_analysis(all)文件夹中

    for j in range(50):
        img = cv2.GaussianBlur(image, (3, 3), 3)
        image = img
    path_3 = "images_analysis_all/" + path + "_GaussianBlur" + ".jpg"
    i += 1
    cv2.imwrite(path_3, image)

    # 将原图经过50次高斯模糊后写入到images_analysis(all)文件夹中

img_path = glob.glob("images_analysis_all/*.jpg")

# 循环输入图像，将其转换为灰度，利用拉普拉斯方法对输入图像的方差计算图像的聚焦测度

for a_path in img_path:
    path = a_path[20:].split('.')[0]

    image = cv2.imread(a_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm > 100.0:
        text = "distinct"
        Img_Distinct = "images_analysis_clear/" + path + ".jpg"
        cv2.imwrite(Img_Distinct, image)

    # 如果对焦测量值小于提供的阈值，则应认为图像'模糊'

    else:
        text = "blur"


    # 照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细

    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    Img_Analyse = "threshold_analysis/" + path + ".jpg"
    cv2.imwrite(Img_Analyse, image)
