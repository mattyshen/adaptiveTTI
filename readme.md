This is an evolving repo optimized for machine-learning projects aimed at designing a new algorithm. They require sweeping over different hyperparameters, comparing to baselines, and iteratively refining an algorithm. Based of [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science).

# Distilling a Model Walk-through
1. Make a copy of `experiments/distillation_starter.py` in the `experiments` folder.
2. Fill out "### TODO: ... ###" in the copy file.
    - `predict_teacher` returns the teacher's prediction
    - `load_teacher_model` returns the loaded in teacher model from the path (models can be stored in the `models` folder).
    -  `generate_tabular_distillation_data` returns the teacher predicted concept design matrices, the true concept design matrices, the predicted teacher outputs (often logits), and true outputs (often classes). Data can be stored in the `data` folder.
    - `process_distillation_data` returns the data the distiller is trained on (often we need to binarize the teacher model's prediction for the distiller model)
    - `process_distiller_eval` returns the distiller's modified prediction to match the metric being logged (sometimes we are using a regressor to distill a classifier DL model's logits, but want to log the distiller's classification performance, so we need to convert the distiller's predicted logits to class predictions)
    - `process_teacher_eval` returns the teacher's modified prediction to match the metric being logged (i.e. a classifier DL model outputs logits, but want to log the model's classification performance, so we need to convert the logits to class predictions)
3. Make a copy of `results/distillations_starter.py` in the `results` folder.
3. Make a copy of `scripts/distillations_starter.py` in the `scripts` folder.
4. Fill out distillation experiment parameters (parameters can be found, with descriptions, in the copy of `experiments/distillation_starter.py` file). The experiment parameters, summarized, consist of teacher model path, train and test data paths, distiller (FIGS) hyperparameters, metric being logged, and number of interactions to intervene on. Edit the `save_dir` path in `params_shared_dict` and `script_name` in the `submit_utils.run_args_list(...)` line at the bottom of the script.
5. `cd` into the scripts folder and run the script.
6. Make a copy of `notebooks/distillations_starter.ipynb` in the `notebooks` folder. Edit the paths in `results_dir = '../results/distillation_cub'` and `experiment_filename = '../experiments/distillation_cub.py'` lines
7. Observe and investigate the results in the `.ipynb` file.


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
