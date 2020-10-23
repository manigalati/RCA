# -*- coding: utf-8 -*-
import sys
import imp
import os
import nibabel as nib
import subprocess

model=#
subprocess.call("mkdir '/content/drive/My Drive/tesi/paper/atlas/"+model+"'", shell=True)
for patient_test in os.listdir("data/testing/"):
  if(os.path.isdir("data/testing/"+patient_test)):
    if(patient_test not in os.listdir("/content/drive/My Drive/tesi/paper/atlas/"+model)):
      subprocess.call("mkdir '/content/drive/My Drive/tesi/paper/atlas/"+model+"/"+patient_test+"'", shell=True)
    frames_test=[]
    for f in os.listdir("data/testing/"+patient_test):
      if("frame" in f and "_gt" not in f):
        frames_test.append(int(f.split("frame")[1].split(".nii.gz")[0]))
    for phase in ["ED","ES"]:
      if(phase=="ED"):
        image_test="data/testing/"+patient_test+"/"+patient_test+"_frame{:02d}.nii.gz".format(min(frames_test))#
        label_test="data/predictions/"+model+"/"+patient_test+"_ED.nii.gz"#
      else:
        image_test="data/testing/"+patient_test+"/"+patient_test+"_frame{:02d}.nii.gz".format(max(frames_test))#
        label_test="data/predictions/"+model+"/"+patient_test+"_ES.nii.gz"#
      for patient_train in os.listdir("data/training/"):
        if(os.path.isdir("data/training/"+patient_train)):
        
          if(patient_train not in os.listdir("/content/drive/My Drive/tesi/paper/atlas/"+model+"/"+patient_test)):
            subprocess.call("mkdir '/content/drive/My Drive/tesi/paper/atlas/"+model+"/"+patient_test+"/"+patient_train+"'", shell=True)
          elif(len(os.listdir("/content/drive/My Drive/tesi/paper/atlas/"+model+"/"+patient_test+"/"+patient_train))<4 and phase=="ED"):
            continue
          elif(len(os.listdir("/content/drive/My Drive/tesi/paper/atlas/"+model+"/"+patient_test+"/"+patient_train))==4):
            continue
          
          frames_train=[]
          for f in os.listdir("data/training/"+patient_train):
            if("frame" in f and "_gt" not in f):
              frames_train.append(int(f.split("frame")[1].split(".nii.gz")[0]))
          if(phase=="ED"):
            image_train="data/training/"+patient_train+"/"+patient_train+"_frame{:02d}.nii.gz".format(min(frames_train))#
            label_train="data/training/"+patient_train+"/"+patient_train+"_frame{:02d}_gt.nii.gz".format(min(frames_train))#
          else:
            image_train="data/training/"+patient_train+"/"+patient_train+"_frame{:02d}.nii.gz".format(max(frames_train))#
            label_train="data/training/"+patient_train+"/"+patient_train+"_frame{:02d}_gt.nii.gz".format(max(frames_train))#


          subprocess.call("cardiovasc_utils -i "+label_test+" --ith 1 3 --dil 3 -o out/label_bin.nii.gz >/dev/null", shell=True)

          #Step 1:
          subprocess.call("reg_aladin -platf 1 -target "+image_test+" -source "+image_train+" -tmask out/label_bin.nii.gz --rigOnly --aff out/affine_1.txt -ln 1 >/dev/null", shell=True)

          #Step 2: Obtain inverse transform
          subprocess.call("reg_transform -target "+image_test+" -invAffine out/affine_1.txt out/inv_affine_1.txt >/dev/null", shell=True)

          #Step 3: Move test to train space
          subprocess.call("reg_resample -target "+image_train+" -source out/label_bin.nii.gz -aff out/inv_affine_1.txt -res out/mask_1.nii.gz -NN >/dev/null", shell=True)

          subprocess.call("cardiovasc_utils -i out/mask_1.nii.gz --dil 7 -o out/mask_2.nii.gz >/dev/null", shell=True)
          
          #Step 4: Move test to train space
          subprocess.call("reg_aladin -platf 1 -source "+image_test+" -tmask out/mask_2.nii.gz -target "+image_train+" -rigOnly -aff out/test_to_train.txt -ln 2 -maxit 20 -fmask out/label_bin.nii.gz -inaff out/inv_affine_1.txt >/dev/null", shell=True)

          subprocess.call("reg_resample -target "+image_train+" -source out/label_bin.nii.gz -aff out/test_to_train.txt -res out/mask_3.nii.gz -NN >/dev/null", shell=True)

          subprocess.call("cardiovasc_utils -i out/mask_3.nii.gz --dil 3 -o out/mask_4.nii.gz >/dev/null", shell=True)
          
          #Step 5: refined registration
          subprocess.call("reg_f3d -ref "+image_train+" -rmask out/mask_4.nii.gz -flo "+image_test+" -cpp out/nrr.cpp.nii -aff out/test_to_train.txt -ln 5 -lp 5 -be 0.1 -jl 0.01 -noAppJL >/dev/null", shell=True)

          #Step 6: Final result
          subprocess.call("reg_resample -ref "+image_train+" -source "+image_test+" -cpp out/nrr.cpp.nii -res '/content/drive/My Drive/tesi/paper/atlas/"+model+"/"+patient_test+"/"+patient_train+"/image_"+phase+".nii.gz' >/dev/null", shell=True)
          subprocess.call("reg_resample -ref "+image_train+" -source "+label_test+" -cpp out/nrr.cpp.nii -res '/content/drive/My Drive/tesi/paper/atlas/"+model+"/"+patient_test+"/"+patient_train+"/seg_"+phase+".nii.gz' >/dev/null", shell=True)