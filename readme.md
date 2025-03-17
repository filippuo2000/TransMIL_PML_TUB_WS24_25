<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="./imgs/transmil_gif.gif" alt="Logo" width="600" height="400">
  </a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      </ul>
    </li>
    <li>
      <a href="#getting-started-with-histopathology">Getting Started with Camelyon16 Dataset (Histopathology)</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#camelyon16-dataset-curation">Camelyon16 Dataset curation</a></li>
        <li><a href="#camelyon16-model-training">Camelyon16 Model Training</a></li>
        <li><a href="#camelyon16-visualization">Camelyon16 Visualization</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->
## Getting Started

This repository is the result of work for the Machine Learning Project course at TUB(Winter Semester 2024/25). The goal of the project was to replicate the results of the ["TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification"](https://arxiv.org/abs/2106.00908), a study that leverages the use of the transformers and Multiple Instance Learning methods to enhance the classification on high-resolution input data slides, which are independently divided into smaller patches. MIL is a widely used approach in weakly supervised learning and very suitable approach to deal with the challenges of the dataset.

In the end the original results not only have been matched, but also outperformed as shown in the table below. This is most likely due to the use of a more data specific feature extraction method - [CTransPath paper] (https://www.semanticscholar.org/paper/Transformer-based-unsupervised-contrastive-learning-WangYang/439e78726a9c4a30216ebc43a82e44758a5a4619) - than the one used in the original paper (pretrained network on ImageNet).

## Test Results Comparison
The final selected model outperforms the model provided by the authors of the TransMIL paper.

| Name                     | Recall (%) | AUC (%)  | Acc (%)  | Specificity (%) |
|--------------------------|-----------|---------|---------|----------------|
| 512_no_ppeg_larger_lr   | 89.8  | **95.00** | **91.47** | 92.50          |
| TransMIL paper model    | n/a       | 93.09   | 88.37   | n/a            |

## xAI
To further analyze the model’s performance, 3 different xAI methods have been implemented - Attention Rollout, Salient Gradients, Integrated Gradients. They were utilized to generate the per-patch importance scores, which allowed for the generation of visual heatmaps presenting where the model focuses its attention in the classification process. Besides the heatmap, a quantitative summary of the model’s decision process has also been obtained.

### Prerequisites

All neccesary dependencies have been listed in the '''sh requirements.txt''' file.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/filippuo2000/TransMIL_PML_TUB_WS24_25.git
   ```
2. Get your wandb key at [https://wandb.ai/site/](https://wandb.ai/site/)
3. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->

Due to the use of very specific features, which are not shared publicly, the training results will not be reproducible. Features have been extracted with the method described in the [CTransPath paper] (https://www.semanticscholar.org/paper/Transformer-based-unsupervised-contrastive-learning-WangYang/439e78726a9c4a30216ebc43a82e44758a5a4619). Their generation was however not a part of the project, as they were provided by its coordinator.
_For usage examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
* Filip Matysik - f.matysik@campus.tu-berlin.de
* 
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Helpful libraries and papers used in the project

* [WandB](https://wandb.ai/site)
* [TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2106.00908)
* [Camelyon Dataset](https://camelyon16.grand-challenge.org/Data/)
* [CTransPath for feature extraction] (https://www.semanticscholar.org/paper/Transformer-based-unsupervised-contrastive-learning-Wang Yang/439e78726a9c4a30216ebc43a82e44758a5a4619)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
## About

## xAI
To further analyze the model’s performance, 3 different xAI methods have been implemented - Attention Rollout, Salient Gradients, Integrated Gradients. They were utilized to generate the per-patch importance scores, which allowed for the generation of visual heatmaps presenting where the model focuses its attention in the classification process. Besides the heatmap, a quantitative summary of the model’s decision process has also been obtained.

## Training and Inference
Checkpoint for the best model has been shared. Below are the instructions on how to use it:

**To run the training please run the below command in the /home/pml16/ folder**

sample_training: <br>
**'apptainer run --nv -B /home/space/datasets/camelyon16:/mnt ./captum_container.sif python MS3/train.py --split /home/pml16/camelyon16_mini_split.csv --run_name mini_run_1'**

full training: <br>
**'apptainer run --nv -B /home/space/datasets/camelyon16:/mnt ./captum_container.sif python MS3/train.py --split /mnt/splits/camelyon16_tumor_85_15_orig_0.csv --config MS3/CamelyonConfig/config.yaml --run_name full_train_1'**

test: <br>
**'apptainer run --nv -B /home/space/datasets/camelyon16:/mnt ./captum_container.sif python MS3/test.py'**

where:
- **-B** mounts the directory with the dataset to the container, so that it can be accesed by apptainer from the inside. It is important to mount this exact directory: /home/space/datasets/camelyon16:/mnt
- **--split** is the absolute path to the chosen split version of the CAMELYON16 dataset. It can be omitted, then the default split is chosen: 'camelyon16_tumor_85_15_orig_0.csv'
- **--config** is the absolute path to the chosen comfiguration of the model and run file, in the .yaml format. It can be omitted, then the default split is chosen: 'config.yaml'
- **--run_name** name of the current experiment, which will be saved to wandb


## xAI - heatmap visualization
**To run the heatmap visualization please run the below command in the /home/pml16/ folder**

heatmap visualization: <br>
**'apptainer run --nv -B /home/space/datasets/camelyon16:/mnt ./captum_container.sif python MS3/visualize_heatmap.py --config MS3/CamelyonConfig/config.yaml --case_id test_027 --method att_rollout --save_dir /home/pml16/'**

where:
- **-B** mounts the directory with the dataset to the container, so that it can be accesed by apptainer from the inside. It is important to mount this exact directory: /home/space/datasets/camelyon16:/mnt
- **--split** is the absolute path to the chosen split version of the CAMELYON16 dataset. It can be omitted, then the default split is chosen: 'camelyon16_tumor_85_15_orig_0.csv'
- **--config** is the absolute path to the chosen comfiguration of the model and run file, in the .yaml format. It can be omitted, then the default split is chosen: 'config.yaml'
- **--case_id** is the id of the test case for which the heatmap will be generated. It can be omitted, then the default sample is: 'test_001'
- **--method** is the absolute path to the chosen explainability method, based on which the heatmap will be visualized. It can be omitted, then the default method is: 'att_rollout'. Other options are: "integrated_grads" and "saliency_grads"
- **--save_dir** is the directory where the heatmap will be saved (it can't). It can be omitted, then the default sample is: 'test_001'
