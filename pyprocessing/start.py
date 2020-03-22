'''
@Date: 2020-02-25 18:38:11
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-04 16:51:38
@Description:  
'''

import patient as pt
import loadpath as lp
import selectset
import os
import gc
from processutils import Logger

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if __name__ == '__main__':
    pathFileName = 'ctpath'
    pathsFilePkl = os.path.join(CURRENT_PATH, pathFileName + '.pkl')
    patientPathSet, setLen = lp.loadPath(pathsFilePkl)
    loggerMain = Logger('mainProcess').getLog
    pList = selectset.selectSet()

    for name in patientPathSet:
        if name in pList:
            loggerMain.info('%s: Start preprocessing.' % name)
            patient = pt.Patient(name, patientPathSet[name])
            patient.start()
            loggerMain.info('%s: Preprocessing completed.' % name)
        else:
            pass