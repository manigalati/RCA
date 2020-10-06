import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from medpy.metric import binary
import operator

def get_results(prediction,reference):
  results={}
  for c,key in enumerate(["","RV_","MYO_","LV_"]):
    ref=np.copy(reference)
    pred=np.copy(prediction)

    ref=ref if c==0 else np.where(ref!=c,0,ref)
    pred=pred if c==0 else np.where(np.rint(pred)!=c,0,pred)
    
    results[key+"SED"]=np.sum((ref-pred)**2)
    results[key+"SED_rint"]=np.sum((ref-np.rint(pred))**2)
    results[key+"maxSED"]=np.sum(((ref-pred)**2).reshape(ref.shape[0],-1),axis=1).max()
    results[key+"maxSED_rint"]=np.sum(((ref-np.rint(pred))**2).reshape(ref.shape[0],-1),axis=1).max()
    results[key+"Dice"]=2*np.sum(pred*np.where(ref!=0,1,0))/np.sum(pred+np.where(ref!=0,1,0)) if np.sum(pred+np.where(ref!=0,1,0))!=0 else 0
    results[key+"Dice_rint"]=binary.dc(np.where(ref!=0,1,0),np.where(np.rint(pred)!=0,1,0))
    try:
      results[key+"HD_rint"]=binary.hd(np.where(ref!=0,1,0),np.where(np.rint(pred)!=0,1,0))
    except Exception:
      results[key+"HD_rint"]=0
  return results

results_dir="/content/drive/My Drive/atlas/"

results = {}
for patient_test in os.listdir(results_dir):
  if(os.path.isdir(results_dir+patient_test)):
    print(patient_test)
    max_dice=0; reference=0; prediction=0;
    for patient_train in os.listdir(results_dir+patient_test):
      gtl = "data/training/"+patient_train+"/"+patient_train+"_frame01_gt.nii.gz"
      outputl = results_dir+patient_test+"/"+patient_train+"/"+"seg.nii.gz"

      gt = nib.load(gtl).get_data()
      pred = nib.load(outputl).get_data()

      dice=binary.dc(np.where(gt!=0,1,0),np.where(np.rint(pred)!=0,1,0))
      if(dice>=max_dice):
        max_dice=dice
        reference=gt
        prediction=pred
      
    results[patient_test]=get_results(prediction,reference)

np.save('RCA_ED.npy',results)
