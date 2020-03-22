'''
@Date: 2020-02-23 16:50:03
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-22 23:50:14
@Description:  
'''
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import pydicom
from skimage import morphology
from skimage.morphology import square
 
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
IMAGEBASKET = os.path.join(CURRENT_PATH, 'imageBasket')
pass

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

def on_mouse(event, x,y, flags , params):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + 'Point' + '('+str(x) + ', ' + str(y)+')')
        seed0 = Point(x, y)
        seeds = [seed0]
        binaryImg = regionGrow(img,seeds,5)
        binaryImg = binaryImg * 255
        cv2.imwrite('Imgs_/R004_2020010_('+str(x)+','+str(y)+').jpg' , binaryImg)
        print('finished' + 'Imgs_/R004_2020010_('+str(x)+','+str(y)+').jpg')
        print(binaryImg)
        cv2.imshow('image',binaryImg)

def dicomPath2Img(dcmPixel, gray=True, path=None):

    if path is None:
        path = os.path.join(IMAGEBASKET, 'cropTraining.jpg')
    cv2.imwrite(path , dcmPixel)
    if gray is True:
        # Gray
        imgPixel = cv2.imread(path, 0)
    else:
        # Three Channels
        imgPixel = cv2.imread(path)
    return imgPixel

def imgStack(img1_1channel, img2_3channel, img3_1channel):
    imgUp = img1_1channel
    imgMiddle = img2_3channel
    imgDown = img3_1channel

    imgMiddle[:,:,0] = imgUp
    imgMiddle[:,:,2] = imgDown

    return imgMiddle

def cropRgb(img, centroid, cropSize, order=False):
    imgUp = img[:,:,0]
    imgMiddle = img[:,:,1]
    imgDown = img[:,:,2]
    imgUp = cropByCentroid(imgUp, centroid, cropSize, order=False)
    imgMiddle = cropByCentroid(imgMiddle, centroid, cropSize, order=False)
    imgDown = cropByCentroid(imgDown, centroid, cropSize, order=False)
    _crop = np.zeros((cropSize,cropSize,3), np.uint8)
    _crop[:,:,0] = imgUp
    _crop[:,:,1] = imgMiddle
    _crop[:,:,2] = imgDown
    return _crop

def cropByCentroid(img, centroid, cropSize, order=False):
    if order is False:
        cy, cx = centroid
    else:
        cx, cy = centroid
    cy = int(cy)
    cx = int(cx)
    y,x = img.shape
    startx = cx-(cropSize//2)
    starty = cy-(cropSize//2)    
    return img[starty:starty+cropSize,startx:startx+cropSize]

def isNoudle(centroid, annos):
    nflag = False
    nType = 'health'
    for noduleType in annos:
        for n in annos[noduleType]:
            nflag = isInrange(centroid, n['centroid'])
            if nflag == True:
                nType = noduleType
                break
    return nType

def isInrange(centroid, noduleCentroid, size=64):
    flag = False
    x = centroid[0]
    y = centroid[1]
    nx = noduleCentroid[0]
    ny = noduleCentroid[1]
    if x > nx - size and x < nx + size and y > ny - size and y < ny + size:
        flag = True
    return flag

def seg_kmeans_gray(graySrc, type = 0):
    k = 5
    if type == 1:
        img = graySrc
    else:
        img = cv2.imread(graySrc, cv2.IMREAD_GRAYSCALE)
 
    img_flat = img.reshape((img.shape[0] * img.shape[1], 1))
    img_flat = np.float32(img_flat)
 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS
 
    #  K-Means Lung Noudel Identify
    compactness, labels, centers = cv2.kmeans(img_flat, k, None, criteria, 10, flags)
    
    img_output = labels.reshape((img.shape[0], img.shape[1]))

    centersSorted = sorted(centers)
    # center1D = centers.reshape((centers.shape[0] * centers.shape[1], 1))
    classNodule = np.where(centers == centersSorted[-1])[0]
    classNoduleEdge = np.where(centers == centersSorted[-2])[0]
    nodulePixels = img_flat[labels==classNodule]
    noduleEdgePixels = img_flat[labels==classNoduleEdge]
    nodulePixelsRange = [min(nodulePixels), max(nodulePixels)]
    noduleEdgePixelsRange = [min(noduleEdgePixels), max(noduleEdgePixels)]
    
    return img_output, nodulePixelsRange, noduleEdgePixelsRange
 

 

