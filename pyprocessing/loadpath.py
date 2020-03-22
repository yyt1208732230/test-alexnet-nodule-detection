'''
@Date: 2020-02-22 17:53:30
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-22 23:03:49
@Description:  LIDC-IDRI Dataset has different types of medical image, including CT, CR, SEG and DX.
               This file help loading and prestore the path of the set from various patients in specify image type for image pre-processing.
'''

import numpy as np
import logging
import os, time
import pickle
import sys
import pydicom

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
IMAGEBASKET = os.path.join(CURRENT_PATH, 'imageBasket')
PYLUNG_PATH = os.path.join(ROOT_PATH, 'pylung')
sys.path.append(PYLUNG_PATH)

import processutils
from processutils import Logger
import dicom_set as dcmset

'''
@description: Save path info to .text and .pkl for each patients in LIDC-IDRI dataset.
@param {type} 
@return: 0
'''
def getPathCT(datasetPath, overwrite=False):
    assert os.path.isdir(datasetPath)
    
    pathFileName = 'ctpath'
    pathsFilePkl = os.path.join(CURRENT_PATH, pathFileName + '.pkl')
    pathsFileTxt = os.path.join(CURRENT_PATH, pathFileName + '.txt')
    logger = Logger('getpath').getLog

    patientPathSet = {}
    logger.info('Traversal of CT Set Begin. ')
    level1_patientset =  os.listdir(datasetPath)
    patientCount = len(level1_patientset)
    logger.info('Patient Count is %d' % patientCount)

    pathCount = 0
    with open(pathsFileTxt,'wt') as fileout:
        # Create dictionary of patients' path 
        for index, setName in  enumerate(level1_patientset):
            tag = False
            if setName.startswith('LIDC-IDRI'):
                writeCount = 0
                # l1path : /LIDC-IDRI/LIDC-IDRI-0163/
                l1path = os.path.join(datasetPath, setName)
                level2_sourceInfo = os.listdir(l1path)
                if len(level2_sourceInfo) > 0:
                    for lsSource in level2_sourceInfo:
                        # l2path : /LIDC-IDRI/LIDC-IDRI-0163/01-01-2000-CT THORAX WCONTRAST-52456/
                        l2path = os.path.join(l1path, lsSource)
                        if(not os.path.isfile(l2path)):
                            level3_equipmentInfo = os.listdir(l2path)
                            for root, dirs, files in os.walk(l2path):
                                for d in dirs:
                                    # l3path : /LIDC-IDRI/LIDC-IDRI-0163/01-01-2000-CT THORAX WCONTRAST-52456/3-Recon 2 CHEST-54230/
                                    l3path = os.path.join(root, d)
                                    dcmSet = os.listdir(l3path)
                                    if len(dcmSet) >= 60:
                                        tag = True
                                        imageCount = 0
                                        pathCount += 1
                                        writeCount += 1
                                        for dcm in dcmSet:
                                            if(dcm.endswith('.dcm')):
                                                imageCount += 1
                                        pathObj = {'ctPath': l3path, 'imageCount': imageCount}
                                        
                                        # Append path into Dictionary
                                        if not setName in patientPathSet:
                                            patientPathSet[setName] = []
                                        patientPathSet[setName].append(pathObj)

                                        # Write path on text file
                                        fileout.write(l3path + '\n')
                                        
                                        # Warning if one patient has more than one CT set
                                        if writeCount > 1:
                                            logging.warning(setName + ': Write count larger than one.')
                                break
                
            if tag == False:
                logging.warning(setName + ': Neither CT nor CR/DX/SEG was wrote in path.')

    # Write paths on pickle file
    if(savePatientObj(pathsFilePkl, patientPathSet)):
        logger.info('Pickel Wrote Completed. Patient count is: %d' % len(patientPathSet))
    else:
        logger.error('Pickel Wrote Error.')

    logger.info('Traversal of CT Set Completed. Path count is: %d' % pathCount)
    return 0

'''
@description: Write CT paths Object on pickle file
@param {type} 
@return: True - if without exception
'''
def savePatientObj(pickleFilePath, data):
    with open(pickleFilePath, 'wb') as f:
        pickle.dump(data, f)
    return True

'''
@description: Load CT path and return an dctionary
@param {type} 
@return: patientPathSet - Dictionary with patients' CT path
         setLen - len of patientPathSet
'''
def loadPath(pickleFilePath):
    pickledict = {}
    setLen = 0
    with open(pickleFilePath, 'rb') as f:
       pickledict = pickle.load(f)
       setLen = len(pickledict)
    return pickledict, setLen

'''
@description: 
@param {type} 
@return: 
'''
def getDicomPath(ctSetPath):
    assert os.path.isdir(ctSetPath)
    
    dicoms = {}
    files = os.listdir(ctSetPath)
    for f in files:
        if(f.endswith('.dcm')):
            _dcmPath = os.path.join(ctSetPath, f)
            _dcm = pydicom.read_file(_dcmPath)
            ct = dcmset.CTImage(_dcm)
            ctDict = ct.getDict()
            ctDict['Path'] = _dcmPath

            basePath = os.path.join(IMAGEBASKET, 'LPT')
            basePath = os.path.join(basePath, ctDict['PatientID'])
            if not os.path.exists(basePath):
                os.mkdir(basePath)
            lptPath = os.path.join(basePath, f.replace('.dcm', '.npy'))
            stackPath = os.path.join(basePath, f.replace('.dcm', '.jpg'))
            ctDict['LptPath'] = lptPath
            ctDict['StackPath'] = stackPath
            dicoms[ctDict['SOPInstanceUID']] = ctDict

    return dicoms

if __name__ == '__main__':
    pathFileName = 'ctpath'
    datasetPath = os.path.join(ROOT_PATH, 'LIDC-IDRI')
    pathsFilePkl = os.path.join(CURRENT_PATH, pathFileName + '.pkl')
    getPathCT(datasetPath)

    # Load Pickle File 'ctpath', check whether file is normal
    patientPathSet, setLen = loadPath(pathsFilePkl)
    for p in patientPathSet:
        pDict = patientPathSet[p]
        print(pDict)
        
    