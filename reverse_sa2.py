# -*- coding: utf-8 -*-
import sys
import imp
import os
import nibabel as nib
import subprocess

subprocess.call("mkdir '/content/drive/My Drive/atlas'", shell=True)
for patient_test in os.listdir("data/testing/"):
  if(os.path.isdir("data/testing/"+patient_test)):
    if(patient_test in os.listdir("/content/drive/My Drive/atlas")):
      continue
    subprocess.call("mkdir '/content/drive/My Drive/atlas/"+patient_test+"'", shell=True)
    image_test="data/testing/"+patient_test+"/"+patient_test+"_frame01.nii.gz"
    label_test="data/testing/"+patient_test+"/"+patient_test+"_ED.nii.gz"
    for patient_train in os.listdir("data/training/"):
      if(os.path.isdir("data/training/"+patient_train)):
        subprocess.call("mkdir '/content/drive/My Drive/atlas/"+patient_test+"/"+patient_train+"'", shell=True)
        image_train="data/training/"+patient_train+"/"+patient_train+"_frame01.nii.gz"
        label_train="data/training/"+patient_train+"/"+patient_train+"_frame01_gt.nii.gz"

        subprocess.call("cardiovasc_utils -i "+label_test+" --ith 1 3 --dil 3 -o out/label_bin.nii.gz >/dev/null", shell=True)
        
        #Step 1:
        subprocess.call("reg_aladin -target "+image_test+" -source "+image_train+" -tmask out/label_bin.nii.gz --rigOnly --aff out/affine_1.txt -ln 1 >/dev/null", shell=True)

        #Step 2: Obtain inverse transform
        subprocess.call("reg_transform -target "+image_test+" -invAffine out/affine_1.txt out/inv_affine_1.txt >/dev/null", shell=True)

        #Step 3: Move test to train space
        subprocess.call("reg_resample -target "+image_train+" -source out/label_bin.nii.gz -aff out/inv_affine_1.txt -res out/mask_1.nii.gz -NN >/dev/null", shell=True)

        subprocess.call("cardiovasc_utils -i out/mask_1.nii.gz --dil 7 -o out/mask_2.nii.gz >/dev/null", shell=True)
        
        #Step 4: Move test to train space
        subprocess.call("reg_aladin -source "+image_test+" -tmask out/mask_2.nii.gz -target "+image_train+" -rigOnly -aff out/test_to_train.txt -ln 2 -maxit 20 -fmask out/label_bin.nii.gz -inaff out/inv_affine_1.txt >/dev/null", shell=True)

        subprocess.call("reg_resample -target "+image_train+" -source out/label_bin.nii.gz -aff out/test_to_train.txt -res out/mask_3.nii.gz -NN >/dev/null", shell=True)

        subprocess.call("cardiovasc_utils -i out/mask_3.nii.gz --dil 3 -o out/mask_4.nii.gz >/dev/null", shell=True)
        
        #Step 5: refined registration
        subprocess.call("reg_f3d -ref "+image_train+" -rmask out/mask_4.nii.gz -flo "+image_test+" -cpp out/nrr.cpp.nii -aff out/test_to_train.txt -ln 5 -lp 5 -be 0.1 -jl 0.01 -noAppJL >/dev/null", shell=True)

        #Step 6: Final result
        subprocess.call("reg_resample -ref "+image_train+" -source "+image_test+" -cpp out/nrr.cpp.nii -res '/content/drive/My Drive/atlas/"+patient_test+"/"+patient_train+"/image.nii.gz' >/dev/null", shell=True)
        subprocess.call("reg_resample -ref "+image_train+" -source "+label_test+" -cpp out/nrr.cpp.nii -res '/content/drive/My Drive/atlas/"+patient_test+"/"+patient_train+"/seg.nii.gz' >/dev/null", shell=True)