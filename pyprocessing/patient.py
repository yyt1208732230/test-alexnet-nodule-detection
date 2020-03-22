'''
@Date: 2020-02-23 17:15:58
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-22 23:20:08
@Description:  Class for a patient what should process.
'''

import os
import gc
import sys
import cv2
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load
from matplotlib import pyplot as plt
import pickle
import pydicom

import loadpath as lp
import imgfactory as imgf
import singleprocess as sgpro
from processutils import Logger

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PYLUNG_PATH = os.path.join(ROOT_PATH, 'pylung')
TRAINING_PATH = os.path.join(ROOT_PATH, 'TrainingSet')
IMAGEBASKET = os.path.join(CURRENT_PATH, 'imageBasket')
sys.path.append(PYLUNG_PATH)
logger = Logger('patientProcess').getLog

import annotation

class Patient:

    def __init__(self, name, metaPatient=[]) :
        self.name = name
        self.metaPatient = metaPatient
        self.caseCount = len(metaPatient)

    def patientProcess(self):
        for case in self.metaPatient:
            self.caseProcess(case)

    def caseProcess(self, case):
        try :
            casePath = case['ctPath']
            pathsFilePkl = os.path.join(casePath, 'dicoms_detail.pkl')
            annotationFlattenPath = os.path.join(casePath, 'annotation_flatten.pkl')
            
            logger.info('%s case process begin.' % self.name)
            # Get dicom details and write on dicoms_detail.pkl
            dicoms = lp.getDicomPath(casePath)
            lp.savePatientObj(pathsFilePkl, dicoms)
            logger.info('%s case details wrote into pickle.' % self.name)

            # Parse Nodule details on annotation.pkl and annotation_flatten.pkl 4 respectively
            annotation.parse_dir(casePath)

            # Load Nodule Detials and Extracting Noduels
            anno = self.loadNoudleInfo(annotationFlattenPath)
            dcmsInfo = self.loadDicomSetInfo(pathsFilePkl)
            if len(anno) == 0 or len(dcmsInfo) == 0:
                logger.warning('%s: Pickle file had something error while read. This case has skipped.' % self.name)
                return
            if len(dcmsInfo) > 200:
                logger.warning('%s: Case too large, CT more than 200. This case has skipped.' % self.name)
                return

            # Processing One: Generate Pulmonary Parenchyma Template (LPT)
            self.lptGenarate(dcmsInfo)
            
            # Generate Training Set
            self.trainingSetGenerate(anno, dcmsInfo)

            # Generate Training Set
            self.stackGenarate(dcmsInfo)
            logger.info('Stack Completed.')
            self.healthySetGenarate(anno, dcmsInfo)
            
        except Exception as e:
            logger.error('{name}: case error - {e}'.format(name=self.name, e=e))
                
    def lptGenarate(self, dcmsInfo):
        for soplpt in dcmsInfo:
            ctDicom = dcmsInfo[soplpt]
            dicomPath = ctDicom['Path']
            pid = ctDicom['PatientID']
            name = str(ctDicom['InstanceNumber'])
            npyPath = ctDicom['LptPath']
            LPT = sgpro.processone(dicomPath, pid, name)

            pltData = asarray(LPT)
            save(npyPath, pltData)
            
        return 0
    
    def trainingSetGenerate(self, annos, dcmsInfo):
        imgCount = 0
        for sop in annos:
            if not sop in dcmsInfo:
                logger.warning('{name}: SOP UID-{sopuid} not in CT Dicom Set.'.format(name=self.name, sopuid=sop))
                continue
            ctCount = len(dcmsInfo)
            ctDicom = dcmsInfo[sop]

            instanceNumber = ctDicom['InstanceNumber']
            lastInstanceNumber = instanceNumber - 1
            nextInstanceNumber = instanceNumber + 1
            if lastInstanceNumber < 1 or nextInstanceNumber > ctCount:
                logger.warning('{name}: SOP UID-{sopuid} can not process images stack.'.format(name=self.name, sopuid=sop))
                continue

            dicomPath = ctDicom['LptPath']
            lastDicomPath = ''
            nextDicomPath = ''
            for dcmSop in dcmsInfo:
                if dcmsInfo[dcmSop]['InstanceNumber'] == lastInstanceNumber:
                    lastDicomPath = dcmsInfo[dcmSop]['LptPath']
                if dcmsInfo[dcmSop]['InstanceNumber'] == nextInstanceNumber:
                    nextDicomPath = dcmsInfo[dcmSop]['LptPath']
                if lastDicomPath != '' and nextDicomPath != '':
                    break
            
            lastDcmPixel = load(lastDicomPath)
            dicomPixel = load(dicomPath)
            nextDcmPixel = load(nextDicomPath)

            anno = annos[sop]
            noduleList = anno['nodules']
            nonNoduleList = anno['non_nodules']
            smallNoduleList = anno['small_nodules']

            for nodule in noduleList:
                imgCount += 1
                _centroid = nodule['centroid']
                cropLastDcm =  imgf.cropByCentroid(lastDcmPixel, _centroid, cropSize=64, order=True)
                cropDcm =  imgf.cropByCentroid(dicomPixel, _centroid, cropSize=64, order=True)
                cropNextDcm =  imgf.cropByCentroid(nextDcmPixel, _centroid, cropSize=64, order=True)

                imgUp = imgf.dicomPath2Img(cropLastDcm)
                imgMiddle = imgf.dicomPath2Img(cropDcm, gray=False)
                imgDown = imgf.dicomPath2Img(cropNextDcm)

                imgForTrain = imgf.imgStack(imgUp, imgMiddle, imgDown)
                savePath = os.path.join(TRAINING_PATH, 'nodule')
                savePath = os.path.join(savePath, self.name + '_' + str(imgCount) + '_' + str(_centroid[0]) + '-' + str(_centroid[1]) + '_0_0.jpg')
                cv2.imwrite(savePath, imgForTrain)
            
            for nodule in nonNoduleList:
                imgCount += 1
                _centroid = nodule['centroid']
                cropLastDcm =  imgf.cropByCentroid(lastDcmPixel, _centroid, cropSize=64, order=True)
                cropDcm =  imgf.cropByCentroid(dicomPixel, _centroid, cropSize=64, order=True)
                cropNextDcm =  imgf.cropByCentroid(nextDcmPixel, _centroid, cropSize=64, order=True)

                imgUp = imgf.dicomPath2Img(cropLastDcm)
                imgMiddle = imgf.dicomPath2Img(cropDcm, gray=False)
                imgDown = imgf.dicomPath2Img(cropNextDcm)

                imgForTrain = imgf.imgStack(imgUp, imgMiddle, imgDown)
                savePath = os.path.join(TRAINING_PATH, 'non_nodule')
                savePath = os.path.join(savePath, self.name + '_' + str(imgCount) + '_' + str(_centroid[0]) + '-' + str(_centroid[1]) + '_0_2.jpg')
                cv2.imwrite(savePath, imgForTrain)

            for nodule in smallNoduleList:
                imgCount += 1
                _centroid = nodule['centroid']
                cropLastDcm =  imgf.cropByCentroid(lastDcmPixel, _centroid, cropSize=64, order=True)
                cropDcm =  imgf.cropByCentroid(dicomPixel, _centroid, cropSize=64, order=True)
                cropNextDcm =  imgf.cropByCentroid(nextDcmPixel, _centroid, cropSize=64, order=True)

                imgUp = imgf.dicomPath2Img(cropLastDcm)
                imgMiddle = imgf.dicomPath2Img(cropDcm, gray=False)
                imgDown = imgf.dicomPath2Img(cropNextDcm)

                imgForTrain = imgf.imgStack(imgUp, imgMiddle, imgDown)
                savePath = os.path.join(TRAINING_PATH, 'small_nodule')
                savePath = os.path.join(savePath, self.name + '_' + str(imgCount) + '_' + str(_centroid[0]) + '-' + str(_centroid[1]) + '_0_1.jpg')
                cv2.imwrite(savePath, imgForTrain)

    def stackGenarate(self, dcmsInfo):
        for dcmsop in dcmsInfo:
            ctCount = len(dcmsInfo)
            ctDicom = dcmsInfo[dcmsop]

            instanceNumber = ctDicom['InstanceNumber']
            lastInstanceNumber = instanceNumber - 1
            nextInstanceNumber = instanceNumber + 1

            dicomPath = ctDicom['LptPath']
            stackPath = ctDicom['StackPath']
            lastDicomPath = ''
            nextDicomPath = ''

            for dcmSop in dcmsInfo:
                if dcmsInfo[dcmSop]['InstanceNumber'] == lastInstanceNumber:
                    lastDicomPath = dcmsInfo[dcmSop]['LptPath']
                if dcmsInfo[dcmSop]['InstanceNumber'] == nextInstanceNumber:
                    nextDicomPath = dcmsInfo[dcmSop]['LptPath']
                if lastDicomPath != '' and nextDicomPath != '':
                    break
            
            if lastInstanceNumber >= 1:
                lastDcmPixel = load(lastDicomPath)
            else:
                lastDcmPixel = load(dicomPath)
                
            if nextInstanceNumber <= ctCount:
                nextDcmPixel = load(nextDicomPath)
            else:
                nextDcmPixel = load(dicomPath)

            dicomPixel = load(dicomPath)

            imgUp = imgf.dicomPath2Img(lastDcmPixel)
            imgMiddle = imgf.dicomPath2Img(dicomPixel, gray=False)
            imgDown = imgf.dicomPath2Img(nextDcmPixel)

            imgForTrain = imgf.imgStack(imgUp, imgMiddle, imgDown)
            _basket = os.path.join(IMAGEBASKET, 'LPT')
            _basket = os.path.join(_basket, self.name)
            cv2.imwrite(stackPath, imgForTrain)

            # logger.info('Stack Completed.')

    def healthySetGenarate(self, annos, dcmsInfo):
        for dcmsop in dcmsInfo:
            ctCount = len(dcmsInfo)
            ctDicom = dcmsInfo[dcmsop]

            dicomPath = ctDicom['LptPath']
            stackPath = ctDicom['StackPath']
            instanceNumber = ctDicom['InstanceNumber']

            lptPixel = load(dicomPath)
            rgbPixel = cv2.imread(stackPath)

            anno = {}
            if dcmsop in annos:
                anno = annos[dcmsop]

            sgpro.processtwo(lptPixel, rgbPixel, anno, self.name, instanceNumber)

    def loadDicomSetInfo(self, path):
        if os.path.isfile(path):
            logger.info("Loading dicom detail from file %s" % path)
            with open(path, 'rb') as f:
                annotations = pickle.load(f)
            return annotations
        else:
            logger.error("Load dicom detail error")
            return {}

    def loadNoudleInfo(self, path):
        if os.path.isfile(path):
            logger.info("Loading annotations from file %s" % path)
            with open(path, 'rb') as f:
                annotations = pickle.load(f)
            return annotations
        else:
            logger.error("Load annotations error")
            return {}

    def start(self):
        logger.info('%s: Patient process start.' % self.name)
        self.patientProcess()
        logger.info('%s: Patient process completed.' % self.name)

if __name__ == '__main__':
    # example
    pathFileName = 'ctpath'
    pathsFilePkl = os.path.join(CURRENT_PATH, pathFileName + '.pkl')
    patientPathSet, setLen = lp.loadPath(pathsFilePkl)
    for name in patientPathSet:
        patient = Patient(name, patientPathSet[name])
        patient.patientProcess()
        break
    