'''
@Date: 2020-02-16 01:17:38
@LastEditors: Laurence Yu
@LastEditTime: 2020-02-25 18:18:54
@Description:  Read DICOM and Information extraction
'''
import os
import numpy as np
import pydicom as pd

class DcmImage:
    def __init__(self, dcm):
        self.modality = dcm.get('Modality')
        self.patient_id = dcm.get('PatientID')
        self.study_id = dcm.get('StudyInstanceUID')
        self.series_uid = dcm.get('SeriesInstanceUID')
        self.sop_uid = dcm.get('SOPInstanceUID')
        self.patient_sex = dcm.get('PatientSex')
        self.patient_age = dcm.get('PatientAge')
        # self.pixels = dcm.pixel_array

    def __str__(self):
        string = ''
        string += 'Modality' + self.modality + "\n"
        string += 'Patient ID' + self.patient_id + "\n"
        string += 'Study UID' + self.study_id + "\n"
        string += 'Series UID' + self.series_uid + "\n"
        string += 'SOP UID' + self.sop_uid + "\n"
        string += 'Patien tSex' + self.patient_sex + "\n"
        string += 'Patient Age' + self.patient_age + "\n"
        return string

    def getDict(self):
        imageDict = {}
        imageDict['Modality'] = self.modality
        imageDict['PatientID'] = self.patient_id
        imageDict['StudyInstanceUID'] = self.study_id
        imageDict['SeriesInstanceUID'] = self.series_uid
        imageDict['SOPInstanceUID'] = self.sop_uid
        imageDict['PatientSex'] = self.patient_sex
        imageDict['PatientAge'] = self.patient_age
        return imageDict

class CTImage:
    def __init__(self, dcm):
        DcmImage.__init__(self, dcm)
        self.z = dcm.get('SliceLocation')
        self.instance_num = dcm.get('InstanceNumber')
    
    def __str__(self):
        str = DcmImage.__str__(self)
        str += 'z:%f\n' % self.z
        str += 'instance_num:%d\n' % self.instance_num
        return str

    def getDict(self):
        imageDict = DcmImage.getDict(self)
        imageDict['SliceLocation'] = self.z
        imageDict['InstanceNumber'] = self.instance_num
        return imageDict

if __name__ == '__main__':
    patientPath = 'd:/MyLab/GraduateProject/LIDC-IDRI/LIDC-IDRI-0256/01-01-2000-CT  CAP  WO CONT-35073/4-Recon 3 C-A-P-08658/'
    dataset = os.listdir(patientPath)
    patientArr = []
    for _dataDicom in dataset:
        if(_dataDicom.endswith('.dcm')):
            _fName = os.path.join(patientPath, _dataDicom)
            _dcm = pd.read_file(_fName)
            patientArr.append(_dcm)
            ct = CTImage(_dcm)
            if ct.sop_uid.endswith('334276986366937900163861106093'):
                print(_dataDicom)
                print(ct.getDict())
    