# Adaptive Test-Time Intervention for Concept Bottleneck Models

Concept Bottleneck Models (CBMs) offer interpretability by predicting human-understandable concepts before the final target, but often sacrifice predictive performance or rely on black-box  mappings from concepts to target (CTT). This contains code for FIGS-BD, a binary distillation of a CBM’s CTT component into a Fast Interpretable Greedy Sum-Tree (FIGS), which maintains strong prediction performance while improving interpretability. FIGS-BD supports adaptive test-time intervention by identifying key binary concept interactions that most influence predictions—allowing practitioners to selectively correct a small number of concepts that significantly boost model performance in realistic, limited-intervention settings.

## Organization
- `data`: contains NLP data of TBM models.
- `experiments`: contains distillation and adaptive test-time intervention (ATTI) with FIGS-BD, Linear, and Random version scripts for various datasets (CUB, TravelingBirds, AGNews, CEBaB).
- `figures`: contains created plots utilized in the paper.
- `idistill`: contains models like FIGS(-BD) and Transformer concept-to-target models for TBMs.
- `notebooks`: contains notebooks to process the results from the distillation and ATTI, as well as notebook versions of ATTI to generate plots utilized in the paper.
- `scripts`: contains files that allow a user to sweep over distillation and ATTI hyperparameters for an experiment in `experiments`.

## Distilling a Model with FIGS-BD and Using ATTI Walk-through
1. Make a copy of `experiments/distillation_starter.py` in the `experiments` folder.
2. Fill out "### TODO: ... ###" in the copy file.
    - `predict_teacher` returns the teacher's prediction.
    - `load_teacher_model` returns the loaded in teacher model from the path (models can be stored in the `models` folder).
    -  `generate_tabular_distillation_data` returns the teacher predicted concept design matrices, the true concept design matrices, the predicted teacher outputs (often logits), and true outputs (often classes). Data can be stored in the `data` folder.
    - `process_distillation_data` returns the data the distiller is trained on (often we need to binarize the teacher model's prediction for the distiller model).
    - `process_distiller_eval` returns the distiller's modified prediction to match the metric being logged (sometimes we are using a regressor to distill a classifier DL model's logits, but want to log the distiller's classification performance, so we need to convert the distiller's predicted logits to class predictions).
    - `process_teacher_eval` returns the teacher's modified prediction to match the metric being logged (i.e. a classifier DL model outputs logits, but want to log the model's classification performance, so we need to convert the logits to class predictions).
3. Make a copy of `results/distillation_starter` in the `results` folder.
4. Make a copy of `scripts/distillations_starter.py` in the `scripts` folder.
    - specify the hyperparameters (including number of ATTIs) to sweep over.
5. Fill out distillation experiment parameters (parameters can be found, with descriptions, in the copy of `experiments/distillation_starter.py` file). The experiment parameters, summarized, consist of teacher model path, train and test data paths, distiller (FIGS) hyperparameters, metric being logged, and number of interactions to intervene on. Edit the `save_dir` path in `params_shared_dict` and `script_name` in the `submit_utils.run_args_list(...)` line at the bottom of the script.
6. `cd` into the scripts folder and run the script.
7. Make a copy of `notebooks/distillation_starter.ipynb` in the `notebooks` folder. Edit the paths in `results_dir = '../results/distillation_starter'` and `experiment_filename = '../experiments/distillation_starter.py'` lines
8. Observe and investigate the results in the `.ipynb` file.

## Citation
If you use any of our code in your work, please cite:
```bash
@misc{shen2025atti,
      title={Adaptive Test-Time Intervention for Concept Bottleneck Models}, 
      author={Matthew Shen and Aliyah Hsu and Abhineet Agarwal and Bin Yu},
      year={2025},
      eprint={2503.06730},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.06730}, 
}
```

## Features
- Based of [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science). See linked repo for an evolving repo optimized for machine-learning projects aimed at designing a new algorithm. They require sweeping over different hyperparameters, comparing to baselines, and iteratively refining an algorithm.
