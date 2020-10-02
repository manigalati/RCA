import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from medpy.metric.binary import dc
import operator

results_dir="/content/drive/My Drive/atlas/"

rankRCA = {}
rankRCA_RV = {}
rankRCA_MYO = {}
rankRCA_LV = {}
for patient_test in os.listdir(results_dir):
  if(os.path.isdir(results_dir+patient_test)):
    print(patient_test)

    dices = []
    dices_RV = []
    dices_MYO = []
    dices_LV = []
    for patient_train in os.listdir(results_dir+patient_test):
      gtl = "data/training/"+patient_train+"/"+patient_train+"_frame01_gt.nii.gz"
      outputl = results_dir+patient_test+"/"+patient_train+"/"+"seg.nii.gz"

      gt = nib.load(gtl).get_data()
      pred = nib.load(outputl).get_data()
      pred = np.rint(pred)

      res={}
      for c,key in enumerate(["RV", "MYO", "LV"],start=1):
        gt_c = np.copy(gt)
        gt_c[gt_c != c] = 0

        pred_c = np.copy(pred)
        pred_c[pred_c != c] = 0

        gt_c = np.clip(gt_c, 0, 1)
        pred_c = np.clip(pred_c, 0, 1)

        res[key] = dc(gt_c, pred_c)
      
      dices_RV.append(res["RV"])
      dices_MYO.append(res["MYO"])
      dices_LV.append(res["LV"])

      gt[gt != 0] = 1
      pred[pred != 0] = 1

      dices.append(dc(gt, pred))
      
    rankRCA[patient_test]=max(dices)
    rankRCA_RV[patient_test]=max(dices_RV)
    rankRCA_MYO[patient_test]=max(dices_MYO)
    rankRCA_LV[patient_test]=max(dices_LV)

rankRCA = dict(sorted(rankRCA.items(), key=operator.itemgetter(1))[::-1])
rankRCA_RV = dict(sorted(rankRCA_RV.items(), key=operator.itemgetter(1))[::-1])
rankRCA_MYO = dict(sorted(rankRCA_MYO.items(), key=operator.itemgetter(1))[::-1])
rankRCA_LV = dict(sorted(rankRCA_LV.items(), key=operator.itemgetter(1))[::-1])

np.save('rankRCA.npy',rankRCA)
np.save('rankRCA_RV.npy',rankRCA_RV)
np.save('rankRCA_MYO.npy',rankRCA_MYO)
np.save('rankRCA_LV.npy',rankRCA_LV)
