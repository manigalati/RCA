from metrics_acdc import *

import os
import numpy as np
import SimpleITK as sitk
import operator

results_dir="results_dir/"#/content/drive/My Drive/atlas/"

pred_DSC = {}
for patient_test in os.listdir(results_dir):
  if(os.path.isdir(results_dir+patient_test)):
    dicemat = []
    print(patient_test)
    for patient_train in os.listdir(results_dir+patient_test):
      gtl = "data/training/"+patient_train+"/"+patient_train+"_frame01_gt.nii.gz"
      outputl = results_dir+patient_test+"/"+patient_train+"/"+"seg.nii.gz"

      gt, _, header = load_nii(gtl)
      pred, _, _ = load_nii(outputl)

      idg=0 #segmentatin class to predict

      zooms = header.get_zooms()
      res = metrics(gt, pred, zooms)

      dice=res[idg*3]

      dicemat.append(dice)
    pred_DSC[patient_test]=max(dicemat)

pred_DSC = dict(sorted(pred_DSC.items(), key=operator.itemgetter(1))[::-1])

print(pred_DSC)