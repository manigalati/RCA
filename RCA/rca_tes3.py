# -*- coding: utf-8 -*-
"""
@author: vvv214
"""
import os
import numpy as np
import SimpleITK as sitk
import operator
from pwnlib.tubes.process import process

results_dir="results_dir/"#/content/drive/My Drive/atlas/"

pred_DSC = {}
sh = process('/bin/sh')
for patient_test in os.listdir(results_dir):
  if(os.path.isdir(results_dir+patient_test)):
    dicemat = []
    print(patient_test)
    for patient_train in os.listdir(results_dir+patient_test):
      gtl = "data/training/"+patient_train+"/"+patient_train+"_frame01_gt.nii.gz"
      outputl = results_dir+patient_test+"/"+patient_train+"/"+"seg.nii.gz"

      idg=0 #segmentatin class to predict

      sh.sendline("python metrics_acdc.py "+gtl+" "+outputl)
      _,output=str(sh.recvline()),str(sh.recvline())
      dice=float(output.split(",")[1+idg*3])

      dicemat.append(dice)
    pred_DSC[patient_test]=max(dicemat)
sh.close()

pred_DSC = dict(sorted(pred_DSC.items(), key=operator.itemgetter(1))[::-1])

print(pred_DSC)