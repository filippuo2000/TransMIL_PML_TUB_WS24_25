
**To run the training please run the below command:**

apptainer run --nv -B /home/space/datasets/camelyon16:/mnt ./pml_update.sif python MS1/train.py --split /home/pml16/camelyon16_mini_split.csv

where:
-B - mounts the directory with the dataset to the container, so that it can be accesed by apptainer from the inside. It is important to mount this exact directory: /home/space/datasets/camelyon16:/mnt
-- split is the absolute path to the chosen split version of the CAMELYON16 dataset. It can be omitted, then the default split is chosen: 'camelyon16_tumor_85_15_orig_0.csv'
