<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="./imgs/transmil_gif.gif" alt="Logo" width="800" height="600">
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

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>
## About

## Test Results Comparison
The final selected model outperforms the model provided by the authors of the TransMIL paper.

| Name                     | Recall (%) | AUC (%)  | Acc (%)  | Specificity (%) |
|--------------------------|-----------|---------|---------|----------------|
| 512_no_ppeg_larger_lr   | 89.8  | **95.00** | **91.47** | 92.50          |
| TransMIL paper model    | n/a       | 93.09   | 88.37   | n/a            |

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
