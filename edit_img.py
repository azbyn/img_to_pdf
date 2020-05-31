import cv2
from math import *
import numpy as np
from matplotlib import pyplot as plt
from sys import argv

def imshow(img, cmap='gray'):
    plt.imshow(img, cmap)
    plt.show()
path = 'p2.jpg' if len(argv) < 2 else argv[1]
out_path = 'res.p2.jpg' if len(argv) < 3 else argv[2]
img = cv2.imread(path)

h, w = img.shape[:2]
if w > h:
    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    #print("ROTATE");
    #center = 
    #M = cv2.getRotationMatrix2D( center, 270, 1.0)
    #img = cv2.warpAffine(img, M, (h, w))

    img = cv2.transpose(img)
    img = cv2.flip(img, 0)
    h, w = img.shape[:2]

    #imshow(img)
    h, w = img.shape[:2]
#imshow(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 9)


canny = cv2.Canny(blurred, 10, 20, 3)

dil = cv2.dilate(canny, None)
res = dil
contours, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
max_approx = []# [[0,0], [w, 0], [w, h], [0, h]]
max_area = 0
for c in contours:
    eps = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    area = abs(cv2.contourArea(approx))
    if approx.shape[0] == 4 and area >= max_area and cv2.isContourConvex(approx):
        max_approx = approx
        max_area = area


out_width = 1000
out_height = out_width * sqrt(2)
dst = np.array([[0, 0], [out_width, 0], [out_width, out_height], [0, out_height], ], dtype=np.float32)
src = np.array([max_approx[0][0], max_approx[1][0], max_approx[2][0], max_approx[3][0]], dtype=np.float32)
def sortCW(arr):
    center = np.array([0, 0], dtype=np.float32)
    for p in arr:
        center += p
    center /= 4
    minAngle = 100
    for it in arr:
        a = atan2(it[1] - center[1], it[0] - center[0])
        if a < minAngle: minAngle = a

    def cmp(it):
        a = atan2(it[1] - center[1], it[0] - center[0])
        return a

    for i in range(4):
        ci = cmp(arr[i])
        for j in range(i+1, 4):
            cj = cmp(arr[j])
            if ci > cj:
                x = arr[i][0]
                y = arr[i][1]
                arr[i] = arr[j]
                arr[j][0] = x
                arr[j][1] = y
    angleCloseEnough = 15 * pi / 180
    #if pi + minAngle < angleCloseEnough:
    #    #0 1 2 3 becomes 1 2 3 0
    #    tempx = arr[0][0]
    #    tempy = arr[0][1]
    #    for i in range(4):
    #        arr[i] = arr[(i+1) % 4]
    #    arr[3][0] = tempx
    #    arr[3][1] = tempy

    return arr


sortCW(dst)
sortCW(src)
# print(dst)
# print(src)
m = cv2.getPerspectiveTransform(src, dst)
res = cv2.warpPerspective(gray, m, (out_width, int(out_height)))
#imshow(res)
cv2.imwrite(out_path, res)
