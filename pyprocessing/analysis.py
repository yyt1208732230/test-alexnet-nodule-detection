'''
@Date: 2020-03-04 16:50:44
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-23 02:20:54
@Description:  Analysis for Data
'''

import patient as pt
import loadpath as lp
import selectset
import os
import gc
from processutils import Logger
import pickle
import pydicom
from matplotlib import pyplot as plt
import cv2
from skimage import io as ioski
import singleprocess as sgp
import imgfactory as imgf
from numpy import load

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def getNoduleCount(path):
    setList = os.listdir(path)
    count = len(setList)
    return count

def getPatientCount(path):
    setList = os.listdir(path)
    patientTotal = len(setList)
    caseCount = 0
    ctCount = 0

    for p in setList:
        ctpath = os.path.join(path, p)
        ctlist = os.listdir(ctpath)
        if not ctpath.endswith('0003'):
            if (len(ctlist) > 0):
                caseCount += 1
            for npy in ctlist:
                if npy.endswith('npy'):
                    ctCount += 1


    return patientTotal, caseCount, ctCount

def testMedian():
    path2= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\dicoms_detail.pkl'
    path1= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\annotation_flatten.pkl'
    step1Path = 'D:\\MyLab\\GraduateProject\\Imgs_\\step1.jpg'
    step2Path = 'D:\\MyLab\\GraduateProject\\Imgs_\\step2-3.jpg'
    step2Path2 = 'D:\\MyLab\\GraduateProject\\Imgs_\\step2-5.jpg'
    step2Path3 = 'D:\\MyLab\\GraduateProject\\Imgs_\\step2-7.jpg'
    step3Path0 = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-0.jpg'
    step3Path = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-3.jpg'
    step3Path2 = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-5.jpg'
    step3Path3 = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-7.jpg'
    f = readfile(path1)
    f2 = readfile(path2)
    # print(f)
    for i in f:
        if i.endswith('1.3.6.1.4.1.14519.5.2.1.6279.6001.334276986366937900163861106093'):
            print(f[i])

    for i2 in f2:
        if f2[i2]['InstanceNumber']==84:
            print(f2[i2])
            pathdic = f2[i2]['Path']

    # _dcm = pydicom.read_file(pathdic)
    # dicomPixel = _dcm.pixel_array
    

    # DICOM source (PIXEL)
    _dcm = pydicom.read_file(pathdic)
    dicomPixel = _dcm.pixel_array
    plt.imshow(dicomPixel,'gray')
    plt.show()

    # Step1 Get Histogram of JPEG pixel (0-256)
    cv2.imwrite(step1Path , dicomPixel)
    pixel1DImg = cv2.imread(step1Path, 0)

    # Step2 Filtering (SMOOTHING)
    pixelForFilteration = pixel1DImg
    img_median = cv2.medianBlur(pixelForFilteration, 3)
    cv2.imwrite(step2Path , img_median)
    img_median = cv2.medianBlur(pixelForFilteration, 5)
    cv2.imwrite(step2Path2 , img_median)
    img_median = cv2.medianBlur(pixelForFilteration, 7)
    cv2.imwrite(step2Path3 , img_median)

    # Step3 Optimal binarization
    pixelForBinarization = cv2.imread(step1Path, 0)
    ret,thresh = cv2.threshold(pixelForBinarization, 0, 255, cv2.THRESH_TRIANGLE)
    ioski.imsave(step3Path0, thresh)
    pixelForBinarization = cv2.imread(step2Path, 0)
    ret,thresh = cv2.threshold(pixelForBinarization, 0, 255, cv2.THRESH_TRIANGLE)
    ioski.imsave(step3Path, thresh)
    pixelForBinarization = cv2.imread(step2Path2, 0)
    ret,thresh = cv2.threshold(pixelForBinarization, 0, 255, cv2.THRESH_TRIANGLE)
    ioski.imsave(step3Path2, thresh)
    pixelForBinarization = cv2.imread(step2Path3, 0)
    ret,thresh = cv2.threshold(pixelForBinarization, 0, 255, cv2.THRESH_TRIANGLE)
    ioski.imsave(step3Path3, thresh)

def testThreshold():
    path2= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\dicoms_detail.pkl'
    path1= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\annotation_flatten.pkl'
    step1Path = 'D:\\MyLab\\GraduateProject\\Imgs_\\step1-.jpg'
    step2Path2 = 'D:\\MyLab\\GraduateProject\\Imgs_\\step2-5-.jpg'
    patha = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-nosmooth.jpg'
    pathb = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-triangle.jpg'
    pathc = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-ostu.jpg'
    pathd = 'D:\\MyLab\\GraduateProject\\Imgs_\\step3-thr128.jpg'

    f = readfile(path1)
    f2 = readfile(path2)
    # print(f)
    for i in f:
        if i.endswith('1.3.6.1.4.1.14519.5.2.1.6279.6001.334276986366937900163861106093'):
            print(f[i])

    for i2 in f2:
        if f2[i2]['InstanceNumber']==84:
            print(f2[i2])
            pathdic = f2[i2]['Path']

    # _dcm = pydicom.read_file(pathdic)
    # dicomPixel = _dcm.pixel_array
    

    # DICOM source (PIXEL)
    _dcm = pydicom.read_file(pathdic)
    dicomPixel = _dcm.pixel_array
    # plt.imshow(dicomPixel,'gray')
    # plt.show()

    # Step1 Get Histogram of JPEG pixel (0-256)
    cv2.imwrite(step1Path , dicomPixel)
    pixel1DImg = cv2.imread(step1Path, 0)

    # Step2 Filtering (SMOOTHING)
    pixelForFilteration = pixel1DImg
    plt.figure("Before Smoothing")
    arr=pixelForFilteration.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red')  
    plt.show()
    ret,thresh = cv2.threshold(pixelForFilteration, 0, 255, cv2.THRESH_TRIANGLE)
    print('Before smoothing : {ret}'.format(ret=ret))
    ioski.imsave(patha, thresh)
    img_median = cv2.medianBlur(pixelForFilteration, 5)
    cv2.imwrite(step2Path2 , img_median)


    # Step3 Optimal binarization
    pixelForBinarization = cv2.imread(step2Path2, 0)
    plt.figure("After Smoothing")
    arr=pixelForBinarization.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red')  
    plt.show()

    ret,thresh = cv2.threshold(pixelForBinarization, 0, 255, cv2.THRESH_TRIANGLE)
    print('TRIANGLE : {ret}'.format(ret=ret))
    ioski.imsave(pathb, thresh)
    pixelForBinarization = cv2.imread(step2Path2, 0)
    ret,thresh = cv2.threshold(pixelForBinarization, 0, 255, cv2.THRESH_OTSU)
    print('THRESH_OTSU : {ret}'.format(ret=ret))
    ioski.imsave(pathc, thresh)
    pixelForBinarization = cv2.imread(step2Path2, 0)
    ret,thresh = cv2.threshold(pixelForBinarization, 128, 255, cv2.THRESH_BINARY)
    print('THR128 : {ret}'.format(ret=ret))
    ioski.imsave(pathd, thresh)

def testKmeans():
    path2= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\dicoms_detail.pkl'
    path1= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\annotation_flatten.pkl'
    dicomPath = 'D:\\MyLab\\GraduateProject\\Imgs_\\256-ins84.npy'

    f = readfile(path1)
    f2 = readfile(path2)
    # print(f)
    for i in f:
        if i.endswith('1.3.6.1.4.1.14519.5.2.1.6279.6001.334276986366937900163861106093'):
            print(f[i])

    for i2 in f2:
        if f2[i2]['InstanceNumber']==84:
            print(f2[i2])
            pathdic = f2[i2]['Path']
            stackPath = f2[i2]['StackPath']

    lptPixel = load(dicomPath)
    rgbPixel = cv2.imread(stackPath)

    sgp.noduleRoiExtract(lptPixel, rgbPixel, 'medianTest', '0')
    
def readfile(path):
    with open(path, 'rb') as f:
        _f = pickle.load(f)
    return _f

if __name__ == '__main__':

    # a, b, c = getPatientCount('D:\\MyLab\\GraduateProject\\pyprocessing\\imageBasket\\LPT')
    # print('Patient Total is: {count}, Case Total is: {count2}, CT count is: {count3}'.format(count=a, count2=b, count3=c))
    
    # path2= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\dicoms_detail.pkl'
    # path1= 'D:\\MyLab\\GraduateProject\\LIDC-IDRI\\LIDC-IDRI-0256\\01-01-2000-CT  CAP  WO CONT-35073\\4-Recon 3 C-A-P-08658\\annotation_flatten.pkl'
    # f = readfile(path1)
    # f2 = readfile(path2)
    # for i in f:
    #     if i.endswith('1.3.6.1.4.1.14519.5.2.1.6279.6001.334276986366937900163861106093'):
    #         print(f[i])

    # for i2 in f2:
    #     if f2[i2]['InstanceNumber']==84:
    #         print(f2[i2])
    #         pathdic = f2[i2]['Path']

    # sgp.processone(pathdic, 'medianTest', '0')

    # testThreshold()

    testKmeans()