
### Interpretability Review

This repository contains code accompanying _Data Science Principles for Interpretable and Explainable AI_. This review article gives an overview of
interpretability research through a statistical lens.

![Summary of XAI Techniques](data/xai-summary.png)

To reproduce the figures in the main text,
you can run these notebooks. None should take longer then 5 minutes to complete.

* `interpretable.Rmd`: Code for the sparse logistic regression and decision tree models on both the raw and featurized versions of microbiome trajectory data. This creates Figure 1 in the text.
* `transformer.ipynb`: Run a transformer on the trajectory data. This trains the model that is analyzed using global embeddings and integrated gradients in the next steps.
* `concept_bottlneck.ipynb`: Run a concept bottleneck model on trajectory data.
* `embeddings.ipynb`: Extract and save the embeddings associated with the trained transformer model.
* `embeddings.Rmd`: Visualize the embeddings saved by `embeddings.ipynb`. This creates Figure 3. 
* `integrated_gradients.ipynb`: Save the integrated gradients for a subset of samples.
* `integrated_gradients.Rmd`: Visualize the integrated gradient estimates saved by `integrated_gradients.ipynb`. This creates Figure 4.

### Data and Environment Setup

The data used in the case study are generated in the notebook
`generate/concept.Rmd`. They are also saved in the the [`data`
folder](https://github.com/krisrs1128/interpretability_review/tree/main/data) of
this repository, in case you want to run the modeling and interpretation code
directly.

If you are running this code on your own laptop, you will need to setup your
environment with the following packages:

* R: `LaplacesDemon`, `RcppEigen`, `broom`, `ggdendro`, `ggrepel`, `glmnetUtils`, `glue`, `gsignal`, `patchwork`, `scico`, `sparsepca`, `tictoc`, `tidymodels`, `tidyverse`
* python: `captum` `lightning`, `numpy`, `pandas`, `tensorboard`, `torch`, `transformer`, 

`LaplacesDemon` and `gsignal` are only needed to regenerate the data, and you
can ignore them if you want to use the files that have already been saved here.
Many of these packages (e.g., `ggrepel`, `ggdendro`, `patchwork`, `scico`) are
only used to refine the visualizations, and you can safely omit them if you are
happy with ggplot2 defaults.

In R, you can install these packages from CRAN:
```
pkgs <- c("LaplacesDemon", "RcppEigen", "broom", "ggdendro", "ggrepel", "glmnetUtils", "glue", "gsignal", "patchwork", "scico", "sparsepca", "tictoc", "tidymodels", "tidyverse")
install.packages(pkgs)
```

For python, you can create a conda environment with these packages:

```
conda create -n interpretability python=3.12
conda activate interpretability

conda install -y conda-forge::lightning
conda install -y conda-forge::pandas
conda install -y conda-forge::tensorboard
conda install -y conda-forge::transformers
conda install -y pytorch::captum
```

All python notebooks are assumed to be run from folder they are saved in, while
the R notebooks are assumed to be run with the repository root
(`interpretability_review`) as the working directory.

### Contact

If you have any questions, don't hesitate to create an issue or reach out to
[ksankaran@wisc.edu](mailto:ksankaran@wisc.edu)
