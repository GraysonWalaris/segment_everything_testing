# Exploring the Segment Anything model for potential data labeling

# Description
This repo holds files for testing and exploring the use of SAM (Segment Anything model from Meta) for labeling data.

# Setup
Clone the repository and install the conda environment by navigating to the home directory of the repository and running

```
conda env create -f environment.yml
```

Download the ViT-H SAM model checkpoint from the original repository here https://github.com/facebookresearch/segment-anything#model-checkpoints and place it in the examples/checkpoints folder. Make sure you rename it to `sam_vit_h.pth` if you want to run the example files with no modification. Alternatively, you can download any other model checkpoint from the original repository, but you will have to manually edit the checkpoint paths and model types of the example python files to do so.

