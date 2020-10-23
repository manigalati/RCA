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
      results[key+"HD_rint"]=np.nan
  return results

model="Isensee"
results_dir="/content/drive/My Drive/tesi/paper/atlas/"+model+"/"

results = {}
for phase in ["ES,ED"]:
  results[phase] = {}
  for patient_test in os.listdir(results_dir):
    if(os.path.isdir(results_dir+patient_test)):
      print(patient_test)
      results[phase][patient_test]={}
      max_dice=0;min_hd=-1;reference_dice=0;prediction_dice=0;reference_hd=0;prediction_hd=0;
      for patient_train in os.listdir(results_dir+patient_test):
        frames_train=[]
        for f in os.listdir("data/training/"+patient_train):
          if("frame" in f and "_gt" not in f):
            frames_train.append(int(f.split("frame")[1].split(".nii.gz")[0]))
        if(phase=="ED"):
          gtl="data/training/"+patient_train+"/"+patient_train+"_frame{:02d}_gt.nii.gz".format(min(frames_train))
          outputl = results_dir+patient_test+"/"+patient_train+"/"+"seg_ED.nii.gz"
        else:
          gtl="data/training/"+patient_train+"/"+patient_train+"_frame{:02d}_gt.nii.gz".format(max(frames_train))
          outputl = results_dir+patient_test+"/"+patient_train+"/"+"seg_ES.nii.gz"
        gt = nib.load(gtl).get_data()
        pred = nib.load(outputl).get_data()
        dice=binary.dc(np.where(gt!=0,1,0),np.where(np.rint(pred)!=0,1,0))
        try:
          hd=binary.hd(np.where(gt!=0,1,0),np.where(np.rint(pred)!=0,1,0))
        except Exception:
          hd=np.nan
        if(dice>=max_dice):
          max_dice=dice
          reference_dice=gt
          prediction_dice=pred
        if(hd<=min_hd or min_hd==-1):
          min_hd=hd
          reference_hd=gt
          prediction_hd=pred
        
      results[phase][patient_test]["Dice"]=get_results(prediction_dice,reference_dice)
      results[phase][patient_test]["HD"]=get_results(prediction_hd,reference_hd)

np.save('RCA.npy',results)
