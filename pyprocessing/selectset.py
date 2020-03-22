'''
@Date: 2020-02-25 23:59:09
@LastEditors: Laurence Yu
@LastEditTime: 2020-02-26 00:26:21
@Description:  Select set for training.
'''

import csv
import os

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def selectSet():
    dignosisPath = os.path.join(ROOT_PATH, 'TCIA_METADATA')
    dignosisPath = os.path.join(dignosisPath, 'tcia-diagnosis-data-2012-04-20.csv')
    processList = []
    with open(dignosisPath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            processList.append(row['PID'])
    return processList

if __name__ == '__main__':
    l = selectSet()
    pass