This is an evolving repo optimized for machine-learning projects aimed at designing a new algorithm. They require sweeping over different hyperparameters, comparing to baselines, and iteratively refining an algorithm. Based of [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science).

# Organization
- `interpretDistill`: main code for modeling (e.g. model architecture)
    - `binary_mapper.py` contains code that discretizes/ binary maps a dataset. This includes the Decision Tree Binary Mapper, the Gaussian Mixture Model Binary Mapper, and the FIGS Binary Mapper.
    - `binary_mapper_utils.py` contains code for utility functions for the Binary Mappers.
    - `continuous.py` keeps a function to determine whether a feature is a continuous feature across multiple files for continuity.
    - `data.py` loads in datasets (currently all regression datasets) for experiments.
    - `figs_d.py` contains a modified copy of the figs file from `imodels` to allow for debugging.
    - `FIGS_nodes.py` contains code to help with FIGS distillation but is not used.
    - `fourierDistill.py` contains the FT Distillation model.
    - `model.py` loads in models for experiments.
    - `params.py` keeps track of model hyperparameters. This file is likely not used in any experiments, though.
    - `subset_predictors.py` contains subset (L0, L0L2) predictor models wrapped in sklearn-like functions.
    - `tabdl.py` contains tabular DL models wrapped in sklearn-like functions.
- `experiments`: code for runnning experiments (e.g. loading data, training models, evaluating models)
    - `06_cv_bm_train_distill_model.py` contains code to train and distill (with FT Distill) a model for a specific dataset with specific hyperparameters.
    - `07_cv_train_model.py` contains code to train a model for a specific dataset with specific hyperparameters.
    - `08_figs_restructure.py` contains code to train a FIGS model for a specific dataset with specific hyperparameters and then restructures the FIGS model with FT Distill.
    - Other `.py` (01-05) files are outdated/not important and the `.ipynb` file contains a notebook for debugging experiment files.
- `scripts`: scripts for hyperparameter sweeps (python scripts that launch jobs in `experiments` folder with different hyperparams)
    - `06_cv_bm_train_distill_models.py` contains code to train and distill (with FT Distill) models across a variety of datasets and hyperparameters.
    - `06_XXX_cv_bm_train_distill_models.py` where `XXX` is `rf`, `tabdl`, `xgb`, or `figs` contains code to train and distill (with FT Distill) models across a variety of datasets and hyperparameters.
    - `07_cv_train_model.py` contains code to train models across a variety of datasets and hyperparameters.
    - `07_XXX_cv_train_models.py.py` where `XXX` is `ftd`, `rf`, `tabdl`, `xgb`, or `figs` contains code to train models across a variety of datasets and hyperparameters.
    - `08_figs_restructures.py` contains code to train FIGS models across a variety of datasets with a variety of hyperparameters, and then restructures the FIGS model with FT Distill.
    - Other `.py` (01-05) files are outdated/not important and the `.ipynb` file contains a notebook for debugging experiment files.
- `notebooks`: jupyter notebooks for analyzing results and making figures
    - `.ipynb` (01-04) files are outdated/not important and the `.ipynb` file contains a notebook for debugging experiment files.
- `tests`: unit tests
- `csv`: contain `.csv` files of results
- `.ipynb` contain exploratory notebook work with some of these models and ideas discussed throughout the summer.

# Features
- scripts sweep over hyperparameters using easy-to-specify python code
- experiments automatically cache runs that have already completed
    - caching uses the (**non-default**) arguments in the argparse namespace
- notebooks can easily evaluate results aggregated over multiple experiments using pandas

# Guidelines
- See some useful packages [here](https://csinva.io/blog/misc/ml_coding_tips)
- Avoid notebooks whenever possible (ideally, only for analyzing results, making figures)
- Paths should be specified relative to a file's location (e.g. `os.path.join(os.path.dirname(__file__), 'data')`)
- Naming variables: use the main thing first followed by the modifiers (e.g. `X_train`, `acc_test`)
    - binary arguments should start with the word "use" (e.g. `--use_caching`) and take values 0 or 1
- Use logging instead of print
- Use argparse and sweep over hyperparams using python scripts (or custom things, like [amulet](https://amulet-docs.azurewebsites.net/main/index.html))
    - Note, arguments get passed as strings so shouldn't pass args that aren't primitives or a list of primitives (more complex structures should be handled in the experiments code)
- Each run should save a single pickle file of its results
- All experiments that depend on each other should run end-to-end with one script (caching things along the way)
- Keep updated requirements in setup.py
- Follow sklearn apis whenever possible
- Use Huggingface whenever possible, then pytorch
