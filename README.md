# Single Image Neural Appearance Estimation

## CONDA ENVIROMENTS
Please use `conda env create -f environment.yml` or `conda env create -f environment_bare.yml`.
`environment_bare.yml` has no version information for dependencies to allow easier version hunting if a version specified in `environment.yml` is no longer available.

Then activate the python environment with `conda activate prlenv`.


## Model Data

Our VGG data for LPIPS perceptual loss is required for training.
Our LDR and HDR model must be downloaded for evaluation and the master demo notebook.

### VGG DATA

Download and untar [vgg_conv.pth](https://drive.google.com/file/d/1hW6cGkmxr1XQPDJ_61atnXwO2yBuJMTZ/view?usp=sharing) into `data/vgg_conv.pth` folder.

### LDR MODEL

Download and untar [ldr_final_prl_model.pth](https://drive.google.com/file/d/1tdnetJpXWmtPkjB5rNCeIahkdLaFD9pJ/view?usp=sharing) into `data/ldr_final_prl_model.pth` folder.


### HDR MODEL

Download and untar [hdr_final_prl_model.pth](https://drive.google.com/file/d/1RE-qpABPiLN0RTxKjR02hGAXRR1VKMUl/view?usp=sharing) into `data/hdr_final_prl_model.pth` folder.



## Datasets

Our training and eval datasets are sourced directly from [MatFusion](https://github.com/samsartor/matfusion)
Ensure you download all three training sets before training.

### INRIA

The inria dataset can be downloaded from https://team.inria.fr/graphdeco/projects/deep-materials/. Unzip it into the `data` directory (`data/DeepMaterialsData/`).
Then run `cd data && python convert_inria.py`. This create a `data/inria_svbrdfs` folder formatted as needed for our training process.

These SVBRDFs are distributed under a CC BY-NC-ND 2.0 licence.

### CC0

Download and untar [cc0_svbrdfs.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/cc0_svbrdfs.tar.lz4) into a `data/cc0_svbrdfs` folder.

These SVBRDFs are collected from PolyHaven and AmbientCG by Sam Sartor for MatFusion, and are distributed under the CC0 licence.

### Mixed

Download and untar [mixed_svbrdfs.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/mixed_svbrdfs.tar.lz4) into a `data/mixed_svbrdfs` folder.

These SVBRDFs are derived from the above INRIA and CC0 datasets by Sam Sartor for MatFusion.

### Test Data

Download and untar [test_svbrdf.tar.lz4](https://www.cs.wm.edu/~ppeers/publications/Sartor2023MFA/data/test_svbrdfs.tar.lz4) into a `data/test_svbrdfs` folder.## Demo

See the prl_master_demo.ipynb for a demonstration of our model (after downloading our LDR model).


## Training

See the prl_main_train.py script for a template of how we train our model.

## Evaluation

See the prl_main_eval.py script for a template of how we evaluate our model.
