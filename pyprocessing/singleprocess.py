'''
@Date: 2020-02-14 11:24:35
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-22 23:42:15
@Description:  This is a single process for one DICOM transfer into specify characteristic values.
'''

import os
import cv2
import datetime
import numpy as np
from matplotlib import pyplot as plt
import imgfactory as imgf
import pydicom as pd
import logging
import time
from skimage import morphology, measure, color
from skimage import io as ioski
from skimage.morphology import square
from processutils import Logger

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
IMAGEBASKET = os.path.join(CURRENT_PATH, 'imageBasket')
TRAINING_PATH = os.path.join(ROOT_PATH, 'TrainingSet')
loggerImg = Logger('imageProcess').getLog

'''
@description: Process for a single DICOM to Pulmonary Parenchyma Template.
@param {type} 
@return: Pixel of Pulmonary Parenchyma Template
'''
def processone(dicomPath, pid, name):
    
    _basket = os.path.join(IMAGEBASKET, 'LPT')
    _basket = os.path.join(_basket, pid)
    step1Path = os.path.join(_basket, name + 'step1.jpg')
    step2Path = os.path.join(_basket, name + 'step2.jpg')
    step3Path = os.path.join(_basket, name + 'step3.jpg')
    step4Path = os.path.join(_basket, name + 'step4.jpg')
    step5Path = os.path.join(_basket, name + 'step5.jpg')
    step6Path = os.path.join(_basket, name + 'step6.jpg')
    step7Path = os.path.join(_basket, name + 'step7.jpg')
    step7Path2 = os.path.join(_basket, name + 'step7_2.jpg')
    step8Path = os.path.join(_basket, name + 'step8.jpg')
    step9Path = os.path.join(_basket, name + 'step9.jpg')

    # DICOM source (PIXEL)
    _dcm = pd.read_file(dicomPath)
    dicomPixel = _dcm.pixel_array

    # Step1 Get Histogram of JPEG pixel (0-256)
    cv2.imwrite(step1Path , dicomPixel)
    pixel1DImg = cv2.imread(step1Path, 0)

    # Step2 Filtering (SMOOTHING)
    pixelForFilteration = pixel1DImg
    img_median = cv2.medianBlur(pixelForFilteration, 5)
    cv2.imwrite(step2Path , img_median)
    loggerImg.debug('Smoothing Completed.')

    # Step3 Optimal binarization
    pixelForBinarization = cv2.imread(step2Path, 0)
    ret,thresh = cv2.threshold(pixelForBinarization, 0, 255, cv2.THRESH_TRIANGLE)
    ioski.imsave(step3Path, thresh)
    loggerImg.debug('Binarization Completed.')

    # Step4 Open operation
    pixelForOpenoper = thresh
    size = 3
    inst = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    img_open=cv2.morphologyEx(pixelForOpenoper,cv2.MORPH_OPEN, inst)
    _pathOpen = step4Path
    cv2.imwrite(_pathOpen , img_open)
    loggerImg.debug('Open Operation Completed.')

    # Step5 Region Growing Operation
    pixelForGroth = img_open
    sp = pixelForGroth.shape
    height = sp[0]  #height(rows) of image
    width = sp[1]   #width(colums) of image
    seed_profile = imgf.Point(3, height-3)
    seed_profile2 = imgf.Point(width-3, 3)
    seeds = [seed_profile, seed_profile2]
    growImg = imgf.regionGrow(pixelForGroth,seeds,5)
    growImg = growImg * 255
    _pathGrow = step5Path
    cv2.imwrite(_pathGrow , growImg)
    loggerImg.debug('Region Growing Operation Completed.')

    # Step6 Get Pulmonary Cavity First Mask by substract operation
    growImg = cv2.imread(_pathGrow, 0)
    openImg = cv2.imread(_pathOpen, 0)
    pulmonaryCavityImg = cv2.absdiff(growImg, openImg)
    cv2.imwrite(step6Path , pulmonaryCavityImg)
    loggerImg.debug('Get Pulmonary Cavity First Mask by Substract Operation Completed.')

    # Step7 Region Growing Operation 2nd and smoothing
    pixelForGroth = pulmonaryCavityImg
    sp = pixelForGroth.shape
    height = sp[0]  #height(rows) of image
    width = sp[1]   #width(colums) of image
    seed_profile = imgf.Point(3, height-3)
    seeds = [seed_profile]
    growImg2 = imgf.regionGrow(pixelForGroth,seeds,9)
    growImg2 = growImg2 * 255
    _pathGrow2 = step7Path
    cv2.imwrite(_pathGrow2 , growImg2)

    growImg2 = cv2.imread(_pathGrow2, 0)
    img_median2 = cv2.medianBlur(growImg2, 17)
    cv2.imwrite(step7Path2 , img_median2)
    # img_median2 = morphology.dilation(img_median2, square(5))
    loggerImg.debug('2nd Region Growing Operation Completed.')

    # Step8 Get reversed Lung Parechyma Temple Mask
    pulmonaryCavityImgNP = np.copy(img_median2)
    pulmonaryCavityPixelsMax = np.max(pulmonaryCavityImgNP)
    pulmonaryCavityMask = pulmonaryCavityPixelsMax - pulmonaryCavityImgNP
    cv2.imwrite(step8Path , pulmonaryCavityMask)

    # lpt = cv2.imread(step9Path, 0)
    # contours, hierarchy = cv2.findContours(lpt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(lpt, contours, -1, (255,255,255), -1)
    # lpt=cv2.morphologyEx(lpt,cv2.MORPH_OPEN, inst)
    # lpt = cv2.medianBlur(lpt, 15)
    # cv2.imwrite(step9Path , lpt)

    # Step9 Get Lung Parechyma Temple Mask by mask operation
    lungParechymaTempImg = cv2.bitwise_and(dicomPixel, dicomPixel, mask=pulmonaryCavityMask)
    cv2.imwrite(step9Path , lungParechymaTempImg)
    loggerImg.debug('Get Lung Parechyma Temple Mask by Substract Operation Completed.')
    # lungParechymaTempImgR = cv2.bitwise_not(lungParechymaTempImg)
    # _pathLPT = os.path.join('Imgs_', 'DICOMTEMP00087' + '_LUNGPARECHYMATEMP' + ".jpg")
    # cv2.imwrite(_pathLPT , lungParechymaTempImg)
    # plt.imshow(lungParechymaTempImg, 'gray')
    # plt.show()
    loggerImg.warning('{pid}-Instance{name}: Get Lung Parechyma Temple Completed.'.format(pid=pid, name=str(name)))
    return lungParechymaTempImg
    
def processtwo(LPT, rgbPixel, anno, pid, instanceName): 

    _basket = os.path.join(IMAGEBASKET, 'LPT')
    _basket = os.path.join(_basket, pid)
    step10Path = os.path.join(_basket, str(instanceName) + 'step10.jpg')
    step11Path = os.path.join(_basket, str(instanceName) + 'step11.jpg')
    # Step1 Keans get threshold
    kms, nodulePixelsRange, noduleEdgePixelsRange = imgf.seg_kmeans_gray(LPT, 1)
    thresoldKmsExtend = min(noduleEdgePixelsRange)
    thresoldKmsCandidateNodule = min(nodulePixelsRange)
    loggerImg.debug('Threshold of K-Means Nodule Identify is %f' % thresoldKmsExtend)

    # Step2 binarization
    retval, dst = cv2.threshold(LPT, thresoldKmsExtend, 255, cv2.THRESH_TOZERO)
    _pathCandidateNodes = step10Path
    cv2.imwrite(_pathCandidateNodes , dst)
    loggerImg.debug('Threshold Operation for Candidate Nodules Completed.')

    # retval, dst2 = cv2.threshold(LPT, thresoldKmsCandidateNodule, 255, cv2.THRESH_TOZERO)
    # _pathCandidateNodes = step11Path
    # cv2.imwrite(_pathCandidateNodes , dst2)
    # loggerImg.debug('Threshold Operation for Candidate Nodules Completed.')

    # Step3 Opening Operaton, Labels and Regionprop
    retval, dst3 = cv2.threshold(dst, 140, 255, cv2.THRESH_BINARY)
    imgRmso = morphology.binary_opening(dst3)
    labels=measure.label(imgRmso,connectivity=1)
    props = measure.regionprops(labels)
    loggerImg.debug('Label Completed, Get props of Label Completed.')

    # Step(optional) Color Labels
    dstColoredB=color.label2rgb(labels)

    # Step4 Remove Odd Tissues
    for p in props:
        if(p['eccentricity']>=0.9):
            clist = p['coords']
            for c in clist:
                yc, xc = c
                labels[yc][xc] = False

    # Color for candidate Nodules
    dstColored=color.label2rgb(labels)
    print('regions number:',labels.max())
    
    # plt.subplot(121), plt.imshow(dstColoredB), plt.title('Before Deleted by Coords')
    # plt.subplot(122), plt.imshow(dstColored), plt.title('Before Deleted by Coords')
    # plt.show()
    
    # Step5 Crop Candidate Nodules
    imgCrops = []
    candidateCount = 1
    for prop in props:
        eccentricity = prop['eccentricity']
        if(eccentricity < 0.9):
            centroid = prop['centroid']
            imgCrops = imgf.cropRgb(rgbPixel, centroid, cropSize=64)
            savePath = os.path.join(TRAINING_PATH, 'healthy_nodule')

            if len(anno) !=0 and imgf.isNoudle(centroid, anno) == 'nodules':
                savePath = os.path.join(TRAINING_PATH, 'nodule')
                _pathCandidateNodeCrops = os.path.join(savePath, pid + '_' + str(candidateCount) + '_' + str(int(centroid[0])) + '-' + str(int(centroid[1])) + '_0_0.jpg')
                cv2.imwrite(_pathCandidateNodeCrops , imgCrops)
            elif len(anno) !=0 and imgf.isNoudle(centroid, anno) == 'small_nodules':
                savePath = os.path.join(TRAINING_PATH, 'small_nodule')
                _pathCandidateNodeCrops = os.path.join(savePath, pid + '_' + str(candidateCount) + '_' + str(int(centroid[0])) + '-' + str(int(centroid[1])) + '_0_1.jpg')
                cv2.imwrite(_pathCandidateNodeCrops , imgCrops)
            elif len(anno) !=0 and imgf.isNoudle(centroid, anno) == 'non_nodules':
                savePath = os.path.join(TRAINING_PATH, 'non_nodule')
                _pathCandidateNodeCrops = os.path.join(savePath, pid + '_' + str(candidateCount) + '_' + str(int(centroid[0])) + '-' + str(int(centroid[1])) + '_0_2.jpg')
                cv2.imwrite(_pathCandidateNodeCrops , imgCrops)
            else:
                savePath = os.path.join(TRAINING_PATH, 'healthy_nodule')
                _pathCandidateNodeCrops = os.path.join(savePath, pid + '_' + str(candidateCount) + '_' + str(int(centroid[0])) + '-' + str(int(centroid[1])) + '_1_3.jpg')
                cv2.imwrite(_pathCandidateNodeCrops , imgCrops)
            # loggerImg.debug('Crop Candidate Nodules No.' + str(candidateCount) + ' Completed. Whichs centroid is ' + str(centroid))
            candidateCount += 1

    loggerImg.warning('Process two Completed.')

def noduleRoiExtract(LPT, rgbPixel, pid, instanceName): 

    _basket = os.path.join(IMAGEBASKET, 'LPT')
    _basket = os.path.join(_basket, pid)
    step10Path = os.path.join(_basket, str(instanceName) + 'step10.jpg')
    step11Path = os.path.join(_basket, str(instanceName) + 'step11.jpg')
    # Step1 Keans get threshold
    kms, nodulePixelsRange, noduleEdgePixelsRange = imgf.seg_kmeans_gray(LPT, 1)
    thresoldKmsExtend = min(noduleEdgePixelsRange)
    thresoldKmsCandidateNodule = min(nodulePixelsRange)
    loggerImg.debug('Threshold of K-Means Nodule Identify is %f' % thresoldKmsExtend)

    # Step2 binarization
    retval, dst = cv2.threshold(LPT, thresoldKmsExtend, 255, cv2.THRESH_TOZERO)
    _pathCandidateNodes = step10Path
    cv2.imwrite(_pathCandidateNodes , dst)
    loggerImg.debug('Threshold Operation for Candidate Nodules Completed.')

    retval, dst2 = cv2.threshold(LPT, thresoldKmsCandidateNodule, 255, cv2.THRESH_TOZERO)
    _pathCandidateNodes = step11Path
    cv2.imwrite(_pathCandidateNodes , dst2)
    loggerImg.debug('Threshold Operation for Candidate Nodules Completed.')

    # Step3 Opening Operaton, Labels and Regionprop
    retval, dst3 = cv2.threshold(dst, 140, 255, cv2.THRESH_BINARY)
    imgRmso = morphology.binary_opening(dst3)
    labels=measure.label(imgRmso,connectivity=1)
    props = measure.regionprops(labels)
    loggerImg.debug('Label Completed, Get props of Label Completed.')

    # Step(optional) Color Labels
    dstColoredB=color.label2rgb(labels)

    # Step4 Remove Odd Tissues
    for p in props:
        if(p['eccentricity']>=0.9):
            clist = p['coords']
            for c in clist:
                yc, xc = c
                labels[yc][xc] = False

    # Color for candidate Nodules
    dstColored=color.label2rgb(labels)
    print('regions number:',labels.max())
    
    plt.subplot(121), plt.imshow(dstColoredB), plt.title('Before Deleted by Coords')
    plt.subplot(122), plt.imshow(dstColored), plt.title('After Deleted by Coords')
    plt.show()
    
    # Step5 Crop Candidate Nodules
    imgCrops = []
    candidateCount = 1
    for prop in props:
        eccentricity = prop['eccentricity']
        if(eccentricity < 0.9):
            centroid = prop['centroid']
            imgCrops = imgf.cropRgb(rgbPixel, centroid, cropSize=64)

            _pathCandidateNodeCrops = os.path.join(_basket, str(instanceName) + '_' + str(candidateCount) + '_' + str(int(centroid[0])) + '-' + str(int(centroid[1])) + '_-_-.jpg')
            cv2.imwrite(_pathCandidateNodeCrops , imgCrops)
            # loggerImg.debug('Crop Candidate Nodules No.' + str(candidateCount) + ' Completed. Whichs centroid is ' + str(centroid))
            candidateCount += 1

    loggerImg.warning('Process two Completed.')

    