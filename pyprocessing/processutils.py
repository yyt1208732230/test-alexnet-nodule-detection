'''
@Date: 2020-02-22 18:42:00
@LastEditors: Laurence Yu
@LastEditTime: 2020-02-25 18:06:36
@Description:  Utils for image processing.
'''

import time
import logging
import os

def getLogsTime():
    currentTime = time.localtime()
    logTimeStr = '' + str(currentTime[0]) + str(currentTime[1]) + str(currentTime[2])
    return logTimeStr

class Logger:
    def __init__(self, prefix='', name=__name__):
        # Set logging visable for each log type.
        logging.basicConfig(level=logging.NOTSET, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        self.__name = name
        self.logger = logging.getLogger(self.__name)
        self.logger.setLevel(logging.NOTSET)

        if prefix != '':
            prefix += '_'
        logFileName = '/log_' + prefix + getLogsTime()+ '.log'
        logPath = os.path.dirname(os.path.abspath(__file__))
        logPath = os.path.join(logPath, 'logs/' + logFileName)
        
        fh = logging.FileHandler(logPath, mode='a', encoding='utf-8')
        fh.setLevel(logging.NOTSET)

        # ch = logging.StreamHandler()
        # ch.setLevel(logging.NOTSET)

        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        # self.logger.addHandler(ch)

    @property
    def getLog(self):
        return self.logger