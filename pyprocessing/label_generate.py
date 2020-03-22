'''
@Date: 2020-02-27 20:44:23
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-23 00:39:33
@Description:  Collect and generate txt(labels) for caffe training within Training Set
'''

import os
import numpy as numpy
import cv2

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
IMAGEBASKET = os.path.join(CURRENT_PATH, 'imageBasket')

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
        if (len(ctlist) > 0):
            caseCount += 1
        for npy in ctlist:
            if npy.endswith('npy'):
                ctCount += 1


    return patientTotal, caseCount, ctCount

def generateTxt():
    setFolder = 'TrainingSet'
    setpath0 = os.path.join(os.path.join(ROOT_PATH, setFolder), 'nodule')
    setpath1 = os.path.join(os.path.join(ROOT_PATH, setFolder), 'small_nodule')
    setpath2 = os.path.join(os.path.join(ROOT_PATH, setFolder), 'non_nodule')
    setpath3 = os.path.join(os.path.join(ROOT_PATH, setFolder), 'health')

    trainingTxt = os.path.join(CURRENT_PATH, 'wholeset.txt')

    ncount = 0
    scount = 0
    nocount = 0
    hcount = 0
    with open(trainingTxt,'wt') as fileout:
        # Nodules
        setList = os.listdir(setpath0)
        for image in setList:
            fileName = 'nodule' + '/' + image
            imgpath = os.path.join(setpath0, image)
            src = cv2.imread(imgpath, 0)
            retval = cv2.countNonZero(src)
            if retval >= 451:
                label = image.split('_', -1)[-2]
                fileout.write(fileName + ' ' + label + '\n')
                ncount += 1
        print('Nodules Generate Completed.')
                
        # Small Nodules
        setList = os.listdir(setpath1)
        for image in setList:
            fileName = 'small_nodule' + '/' + image
            imgpath = os.path.join(setpath1, image)
            src = cv2.imread(imgpath, 0)
            retval = cv2.countNonZero(src)
            if retval >= 451:
                label = image.split('_', -1)[-2]
                fileout.write(fileName + ' ' + label + '\n')
                scount += 1
        print('Small Nodules Generate Completed.')

        # Non Nodules
        setList = os.listdir(setpath2)
        for image in setList:
            fileName = 'non_nodule' + '/' + image
            imgpath = os.path.join(setpath2, image)
            src = cv2.imread(imgpath, 0)
            retval = cv2.countNonZero(src)
            if retval >= 451:
                label = image.split('_', -1)[-2]
                fileout.write(fileName + ' ' + label + '\n')
                nocount += 1
        print('Non Nodules Generate Completed.')
        
        lesionCount = ncount + scount + nocount

        # Health Issues
        setList = os.listdir(setpath3)
        for image in setList:
            fileName = 'health' + '/' + image
            imgpath = os.path.join(setpath3, image)
            src = cv2.imread(imgpath, 0)
            retval = cv2.countNonZero(src)
            if retval >= 451:
                label = image.split('_', -1)[-2]
                fileout.write(fileName + ' ' + label + '\n')
                hcount += 1
        print('Health Issues Generate Completed.')

        print('lesionCount:{lesionCount}, hcount:{hcount}, ncount:{ncount}, scount:{scount}, nocount:{nocount}'.format(lesionCount=lesionCount,hcount=hcount,ncount=ncount,scount=scount,nocount=nocount))
        return lesionCount, hcount, ncount, scount, nocount

def distribute():
    wholeset =os.path.join(CURRENT_PATH, 'wholeset.txt')
    trainingTxt = os.path.join(CURRENT_PATH, 'train.txt')
    valTxt = os.path.join(CURRENT_PATH, 'val.txt')
    testTxt = os.path.join(CURRENT_PATH, 'test.txt')

    readFile = open(wholeset,'rt')
    trainFile = open(trainingTxt,'wt')
    valFile = open(valTxt,'wt')
    testFile = open(testTxt,'wt')

    ncount = 0
    scount = 0
    nocount = 0
    hcount = 0
    lesionCount = 0
    ncount_train = 0
    scount_train = 0
    nocount_train = 0
    hcount_train = 0
    ncount_val = 0
    scount_val = 0
    nocount_val = 0
    hcount_val = 0
    ncount_test = 0
    scount_test = 0
    nocount_test = 0
    hcount_test = 0

    for line in readFile.readlines():
        sp = line.split('_', -1)
        classify = sp[-1].split('.', -1)[0]

        if classify == '0':
            ncount += 1
            n, r = divmod(ncount, 10)
            if r>=1 and r<7 :
                trainFile.write(line)
                ncount_train+=1
            elif r>=7:
                valFile.write(line)
                ncount_val+=1
            else:
                testFile.write(line)
                ncount_test+=1

        if classify == '1':
            scount += 1
            n, r = divmod(scount, 10)
            if r>=1 and r<7 :
                trainFile.write(line)
                scount_train+=1
            elif r>=7:
                valFile.write(line)
                scount_val+=1
            else:
                testFile.write(line)
                scount_test+=1

        if classify == '2':
            nocount += 1
            n, r = divmod(nocount, 10)
            if r>=1 and r<7 :
                trainFile.write(line)
                nocount_train+=1
            elif r>=7:
                valFile.write(line)
                nocount_val+=1
            else:
                testFile.write(line)
                nocount_test+=1

        if classify == '3' and hcount <= lesionCount*1.2:
            hcount += 1
            n, r = divmod(hcount, 10)
            if r>=1 and r<7 :
                trainFile.write(line.replace('non_nodule','health'))
                hcount_train+=1
            elif r>=7:
                valFile.write(line.replace('non_nodule','health'))
                hcount_val+=1
            else:
                testFile.write(line.replace('non_nodule','health'))
                hcount_test+=1

        lesionCount = ncount + scount + nocount

    print(ncount)
    print(scount)
    print(nocount)
    print(hcount)
    print(lesionCount)
    print(ncount_train)
    print(scount_train)
    print(nocount_train)
    print(hcount_train)
    print(ncount_val)
    print(scount_val)
    print(nocount_val)
    print(hcount_val)
    print(ncount_test)
    print(scount_test)
    print(nocount_test)
    print(hcount_test)

if __name__ == '__main__':
    generateTxt()
    distribute()