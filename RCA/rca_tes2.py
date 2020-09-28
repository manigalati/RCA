# -*- coding: utf-8 -*-
"""
@author: vvv214
"""
import os
import numpy as np
import SimpleITK as sitk
import operator

results_dir="results_dir/"#/content/drive/My Drive/atlas/"

pred_DSC = {}
for patient_test in os.listdir(results_dir):
  if(os.path.isdir(results_dir+patient_test)):
    dicemat = []
    for patient_train in os.listdir(results_dir+patient_test):
      gtl = sitk.ReadImage("data/training/"+patient_train+"/"+patient_train+"_frame01_gt.nii.gz")
      outputl = sitk.ReadImage(results_dir+patient_test+"/"+patient_train+"/"+"seg.nii.gz")

      output = sitk.GetArrayFromImage(outputl)
      gt = sitk.GetArrayFromImage(gtl) 

      idg=2 #segmentatin class to predict

      gt[gt != idg] = 0
      gt[gt == idg] = 1
      fmul = np.sum(output  * gt)
      fcom = np.sum(output  + gt)
      if fcom == 0: dice = 0
      else:
          dice  = 2*float(fmul)/float(fcom)
          dice = np.mean(dice)
      dicemat.append(dice)
    pred_DSC[patient_test]=max(dicemat)

pred_DSC = dict(sorted(pred_DSC.items(), key=operator.itemgetter(1))[::-1])

print(pred_DSC)

