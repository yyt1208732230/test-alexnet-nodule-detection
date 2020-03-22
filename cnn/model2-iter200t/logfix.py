'''
@Date: 2020-02-28 20:42:38
@LastEditors: Laurence Yu
@LastEditTime: 2020-02-28 22:38:35
@Description:  
'''

import os
import numpy

if __name__ == '__main__':
    path = './alexnet-nodule-detection-m2-200t.log'
    pathfix = './alexnet-nodule-detection-m2-200t-fix.log'
    ff = open(pathfix, 'w')
    with open(path, 'r') as f:
        newline = ''
        for index ,line in enumerate(f):
            if line == '\n':
                continue
            if index == 0 :
                newline = line
                continue
            head = line.split()
            if len(head)>0 and head[0].startswith('I0'):
                if newline.find('Snapshot') <= -1:
                    ff.write(newline)
                newline = line
            elif head[0] == 'I':
                if newline.strip('\n').endswith(' '):
                    newline = newline.strip('\n')
                    newline += line
                else:
                    if newline.find('Snapshot') <= -1:
                        ff.write(newline)
                    newline = line
            else:
                newline = newline.strip('\n')
                newline += line
        if newline.find('Snapshot') <= -1:
            ff.write(newline)
