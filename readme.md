
**To run the training please run the below command in the /home/pml16/ folder**

sample_training:
apptainer run --nv -B /home/space/datasets/camelyon16:/mnt ./lookahead_container.sif python MS2/train.py --split /home/pml16/camelyon16_mini_split.csv --run_name mini_run_1

full training:
apptainer run --nv -B /home/space/datasets/camelyon16:/mnt ./lookahead_container.sif python MS2/train.py --split /mnt/splits/camelyon16_tumor_85_15_orig_0.csv --run_name full_train_1

where:
- **-B** mounts the directory with the dataset to the container, so that it can be accesed by apptainer from the inside. It is important to mount this exact directory: /home/space/datasets/camelyon16:/mnt
- **--split** is the absolute path to the chosen split version of the CAMELYON16 dataset. It can be omitted, then the default split is chosen: 'camelyon16_tumor_85_15_orig_0.csv'
- **--run_name** name of the current experiment, which will be saved to wandb