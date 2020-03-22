'''
@Date: 2020-03-01 01:06:50
@LastEditors: Laurence Yu
@LastEditTime: 2020-03-23 02:48:49
@Description:  
'''
import os
import numpy

from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, recall_score
import matplotlib.pyplot as plt

def casffe_test():
    listpath = "data/nodulesdetect/test.txt"
    respath = "data/nodulesdetect/test_re.npy"

    testListFile = open(listpath, 'r')

    resultList = {}
    count = 0
    for test in testListFile:
        test = test.split('\n')[0]
        _t = test.split(' ')
        _truth = int(_t[1])
        _jpg = _t[0].replace("/","\\")
        preture = 0
        prefalse = 0
        command = "Build\\x64\\Release\\classification.exe models\\noduledetectmt2\\deploy.prototxt models\\noduledetectmt2\\caffe_alexnet_train_iter_70000.caffemodel data\\nodulesdetect\\mean.binaryproto data\\nodulesdetect\\labels.txt data\\nodulesdetect\\" + _jpg
        result = os.popen(command)  
        res = result.read()  
        for line in res.splitlines():  
            if line.split(' - ')[-1] == '"0 nodule"':
                preture = float(line.split(' - ')[0])
            if line.split(' - ')[-1] == '"1 health-issue"':
                prefalse = float(line.split(' - ')[0])
        resObj = {'truth':_truth, 'pre-Ture':preture, 'pre-False':prefalse}
        resultList[_jpg] = resObj
        count += 1
        if count%50 == 0:
            print count

    print count
    numpy.save(respath, resultList)

    # "0 nodule"
    # "1 health-issue"
    # newres = numpy.load(respath, allow_pickle=True)
    # print(newres)

def drawroc():
    respath = "data/nodulesdetect/test_re.npy"
    newres = numpy.load(respath, allow_pickle=True)
    res = newres.item()
    truth = []
    pre = []
    types = []
    for r in res:
        _type = r.split('.')[-2].split('_')[-1]
        _t = res[r]['truth']
        _p = res[r]['pre-Ture']
        truth.append(_t)
        pre.append(_p)
        types.append(_type)
    nTruth = numpy.array(truth)
    nPre = numpy.array(pre)
    fpr, tpr, thresholds = roc_curve(nTruth, nPre, pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Alexnet Nodule Detection AUC')
    plt.show()

    predit = []
    for p in pre:
        if p > 0.5:
            predit.append(0)
        else:
            predit.append(1)


    total = 0
    noduleHit = 0
    smallHit = 0
    nonHit = 0
    healtHit = 0
    noduleLost = 0
    smallLost = 0
    nonLost = 0
    healtLost = 0

    for i,t in enumerate(types):
        total += 1
        if int(t)==0:
            if truth[i] == predit[i]:
                noduleHit += 1
            else:
                noduleLost += 1
        elif int(t)==1:
            if truth[i] == predit[i]:
                smallHit += 1
            else:
                smallLost += 1
        elif int(t)==2:
            if truth[i] == predit[i]:
                nonHit += 1
            else:
                nonLost += 1
        else:
            if truth[i] == predit[i]:
                healtHit += 1
            else:
                healtLost += 1
    nodulelossR = float(noduleHit)/float(noduleHit+noduleLost)
    smalllossR = float(smallHit)/float(smallLost+smallHit)
    nonlossR = float(nonHit)/float(nonLost+nonHit)

    print('Total={total}\nNodule Hit={noduleHit},Nodule Loss={noduleLost}\nSmall Hit={smallHit},Small Lost={smallLost}\nNon Hit={nonHit},Non Loss={nonLost}\nHealth Hit={healtHit},Health Loss={healtLost}'.format(total=total,noduleHit=noduleHit,noduleLost=noduleLost,smallHit=smallHit,smallLost=smallLost,nonHit=nonHit,nonLost=nonLost,healtHit=healtHit,healtLost=healtLost))
    print('\nnodule loss Rate={nodulelossR},small loss Rate={smalllossR},non loss Rate={nonlossR}'.format(nodulelossR=nodulelossR,smalllossR=smalllossR,nonlossR=nonlossR))
    # accuracy = accuracy_score(truth, predit)
    # print accuracy

    # sensitivity = recall_score(truth, predit, pos_label=0)
    # print sensitivity


if __name__ == "__main__":
    # testing
    casffe_test()

    # get result
    drawroc()