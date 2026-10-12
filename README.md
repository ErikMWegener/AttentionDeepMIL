# Counting wheat ears with attention-based multiple instance learning

Code for the master's thesis *Counting wheat ears on minimally annotated data using
Attention-based Multiple Instance Learning* (Erik Marten Wegener, Martin Luther
University Halle-Wittenberg, Institute for Computer Science, 2026).

Images are cut into patches, and a bag of patches only carries a bag label (does it
contain wheat ears?). Attention-based MIL (AbMIL, Ilse et al. 2018) and a CLAM-based
model (Lu et al. 2021) are trained on these bags. The number of positive patches is
then read out from the attention weights or the instance classifier and converted
into a wheat-ear count.

The repository is a fork of the original AbMIL implementation
([AMLab-Amsterdam/AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)).
The datasets, bag files, MLflow results and trained weights are handed in separately
(USB stick, see [Data](#data)).

```
images (MNIST / GWHD / drone)
   │  data/data_management/create_*_bags.py, mnist_bags_loader_h5.py
   ▼
bag files (HDF5, data/datasets/bags/)
   │  runs/main_mlflow.py   training, evaluation, counting, logging to MLflow
   ▼
MLflow results (runs/mlflow.db, runs/mlruns/)
   │  eval/mlflow_toolbox.py, eval/make_thesis_figures.sh
   ▼
tables and figures (eval/results/)
```

## Repository structure

```
models/
  model.py               AbMIL variants: Attention, AttentionBatchNorm (GroupNorm),
                         AttentionThirdConv, AttentionDropout, GatedAttention
  clam_model.py          CLAM-based model (instance classifier, top-k / quantile pseudo-labels)
  backbone.py            CNN feature extractor shared by AbMIL and CLAM
  learned_grayscale.py   learnt RGB -> grayscale layer
  otsu.py                Otsu threshold for counting with sigmoid / min-max attention
  patch_classifier.py    supervised patch classifier (baseline, backbone pretraining)
  fpn_mil_model.py       FPN-MIL model; implemented and tested but not used in the thesis
data/data_management/
  dataset_manager.py     HDF5 bag format (DatasetWriter / DatasetReader)
  mnist_bags_loader_h5.py   MNIST bags (bag-size sweep and high-density bags)
  create_gwhd_bags.py    GWHD bags
  create_synth_bags.py   bags from the synthetic drone images
  create_drone_bags.py   bags from the time-separated drone images (Drone_BBG)
  create_annot_drone_bags.py   bags from the point-annotated drone tiles (Drone_annot)
  check_dataset.py       count statistics and histograms of a bag dataset
runs/
  main_mlflow.py         entry point: training, test, counting, full-image evaluation
  full_image_count.py    patch count -> ear count on full images
  visualize_features.py  UMAP / t-SNE of the patch features (--visualize_features)
  greedy_search.py       greedy coordinate search over the backbone dimensions
  run_*.sh               experiment scripts (SLURM jobs for the university GPU cluster)
  start_runs.sh          record of the early local runs
  sync_mlflow.sh, import_mlflow.py   copy MLflow results from the cluster into runs/mlflow.db
  relocate_mlflow_db.py  fix the artifact paths after moving runs/mlflow.db
  configs/               base configs used by the run scripts
eval/
  mlflow_toolbox.py, toolbox/   tables and figures from MLflow (see eval/toolbox/README.md, German)
  make_thesis_figures.sh re-creates the figures and tables of the thesis
  scripts/               figure / table scripts and metric functions used by main_mlflow.py
  mnist_bag_sweep/make_density_table.py   table of the MNIST density experiment
  figures/               GWHD example figures (annotation spectrum, counting difficulties)
testing/                 unit tests (pytest)
```

## Installation

The experiments ran in a conda environment on Linux with an NVIDIA GPU (CUDA 12.1).

```bash
conda env create -f environment.yml
conda activate attentiondeepmil
```

Without a GPU, remove the `pytorch-cuda` line from `environment.yml` first; all scripts
then run on the CPU (`main_mlflow.py` also accepts `--no-cuda`).

| Package | Version used | | Package | Version used |
|---|---|---|---|---|
| Python | 3.10.14 | | pandas | 2.0.3 |
| PyTorch | 2.5.1 (CUDA 12.1) | | h5py | 3.15.1 |
| torchvision | 0.20.1 | | Pillow | 12.1.1 |
| NumPy | 1.26.4 | | PyYAML | 6.0.3 |
| SciPy | 1.11.4 | | tqdm | 4.67.3 |
| scikit-learn | 1.6.1 | | umap-learn | 0.5.4 |
| Matplotlib | 3.10.9 | | pytest | 9.0.2 |
| MLflow | 3.13.0 | | OpenCV (headless) | 4.14 locally, 4.10 in `environment.yml` |
| entmax | 1.3 | | | |

`environment.yml` pins the packages that determine the results; the others resolve to
the same or a neighbouring version. OpenCV is only needed for `eval/figures/`.

## Data

Copy the folders from the USB stick into the repository so that the paths below exist.
All scripts use paths relative to the repository.

```
data/datasets/
  MNIST/                       MNIST (as downloaded by torchvision)
  gwhd_2021/                   GWHD 2021: images/ and competition_{train,val,test}.csv
  drone/2025_wheat_heads_datasets/   point-annotated drone tiles (real_data/...) and synthetic images
  drone/2026_wheat_heads_datasets/   drone flights before / after ear emergence (Drone_BBG)
  bags/                        bag files (HDF5)
runs/mlflow.db, runs/mlruns/   MLflow results of all experiments
models/state_dicts/            trained weights (CLAM models of the full-image evaluation,
                               pretrained backbones)
```

After copying `runs/mlflow.db` and `runs/mlruns/`, run `python runs/relocate_mlflow_db.py`
once from the repository root. MLflow stores absolute artifact paths, and the script
points them to the new location (a backup `mlflow.db.bak` is kept).

### Bag datasets

The bag datasets of the thesis (table "Overview of the bag datasets") and where they are:

| Dataset in the thesis | File : group in `data/datasets/bags/` | Created with |
|---|---|---|
| MNIST, bag-size sweep | `mnist_bags_base.h5` : `bags{N}_len{L}` | `mnist_bags_loader_h5.py --mode sweep` |
| MNIST, high density | `mnist_bags_base.h5` : `mnist_bags_even` | `mnist_bags_loader_h5.py --mode even` |
| GWHD, original splits | `gwhd_ogsplit_bags.h5` | `create_gwhd_bags.py --original_split` |
| GWHD, custom splits, grid | `gwhd_bags.h5` : `gwhd_bags` | `create_gwhd_bags.py` |
| GWHD, custom splits, centred | `gwhd_bags_dense.h5` | see [Notes](#notes-and-known-limitations) |
| GWHD, centred, colour | `gwhd_bags_color.h5` | like the centred dataset, without `--grayscale` |
| GWHD, centred, earlier | `gwhd_bags.h5` : `gwhd_bags_dense`; colour variant in `gwhd_bags_2026-05-19.h5` : `gwhd_bags_dense_color` | see Notes |
| GWHD, reduced density | `gwhd_reduced_pos.h5` | see Notes |
| Synthetic | `synth_bags_shuffled.h5` | `create_synth_bags.py` |
| Drone_BBG | `drone_bags.h5` : `drone_128_bags` | `create_drone_bags.py` |
| Drone_annot | `annot_drone_bags.h5` : `annot_drone_{32,64,128}_bags` | `create_annot_drone_bags.py --patch_size {32,64,128}` |
| Drone_annot, full images | raw tiles in `drone/2025_wheat_heads_datasets/real_data/annotated_drone_images_split_4m/` | cut on the fly by `full_image_count.py` |

The generators are run from `data/data_management/` and write to `../datasets/bags/`;
each script describes its options in the module docstring and in `--help`. The MNIST
bags are reproduced bit for bit by `mnist_bags_loader_h5.py`. For all other datasets the
files on the USB stick are the ones used in the thesis; regenerating them is not
guaranteed to give identical bags.

On the cluster the bag files were stored directly in `data/datasets/`; the `path`
parameter of those MLflow runs therefore lacks the `bags/` part.

## Usage

### Training and evaluation

`runs/main_mlflow.py` trains one configuration for one or more seeds, tests it and logs
everything to MLflow (one parent run with the seed aggregates, one child run per seed).
Run it from `runs/`. Use a separate tracking database for new runs, so the submitted
results stay unchanged:

```bash
cd runs
export MLFLOW_TRACKING_URI=sqlite:///mlflow_new.db

# AbMIL on the MNIST high-density bags, five seeds
python main_mlflow.py --path ../data/datasets/bags/mnist_bags_base.h5 --dataset mnist_bags_even \
    --model attention --epochs 20 --seeds 1 10 100 1000 10000 --naive_counting --exp_name demo

# CLAM on the centred GWHD bags with quantile pseudo-labels and a calibrated counting threshold
python main_mlflow.py --path ../data/datasets/bags/gwhd_bags_dense.h5 --dataset gwhd_bags_dense \
    --model clam --epochs 50 --seeds 1 10 100 --clam_pseudo_threshold \
    --clam_pseudo_quantile_pos 0.8 --clam_pseudo_quantile_neg 0.1 --clam_bag_weight 0.7 \
    --count_threshold_eval --calibrate_count_threshold --exp_name demo
```

Main options (`python main_mlflow.py --help` lists all):

| Option | Meaning |
|---|---|
| `--path`, `--dataset` | bag file and group |
| `--model` | `attention`, `gated_attention`, `attention_batchnorm`, `attention_third_conv`, `attention_dropout`, `clam` (`fpn_mil` not used in the thesis) |
| `--attention_activation` | `softmax` (default), `sigmoid`, `min_max`, `sparsemax`, `entmax`, `softmax_temperature` |
| `--model_M`, `--model_L`, `--model_num_maps`, `--model_kernel_size`, `--model_pool_size` | backbone and attention dimensions |
| `--rgb`, `--grayscaling` | colour input, learnt grayscale conversion |
| `--clam_k_sample`, `--clam_bag_weight`, `--clam_pseudo_threshold`, `--clam_pseudo_quantile_pos/neg` | CLAM pseudo-labelling |
| `--naive_counting` | count positive patches with the default rule of the model |
| `--count_threshold_eval`, `--calibrate_count_threshold`, `--soft_counting` | CLAM: compare counting rules, calibrate the threshold on the validation bags |
| `--pretrained_backbone` | load backbone weights from `models/patch_classifier.py` |
| `--save_model`, `--load_model`, `--eval_only` | store / reuse trained weights |
| `--full_image_test`, `--full_image_cal`, `--full_image_only`, `--full_image_patch_size` | full-image evaluation (patch count -> ear count) |
| `--visualize_features`, `--log_attention_weights`, `--log_instance_scores` | additional artifacts |
| `--config` | read the options from a YAML file (runs without `--config` write theirs to `runs/configs/`) |

### Experiments of the thesis

All results of the thesis are stored in `runs/mlflow.db`. The table maps the sections of
the chapter "Experiments and results" to the MLflow experiments and the scripts that
started them. Exact settings are stored as parameters of each run.

| Section | MLflow experiment(s) | Script |
|---|---|---|
| Method validation on MNIST: bag classification, counting, gated attention | `mnist_bag_sweep_5seeds` (first run: `mnist_bag_sweep`) | `runs/run_mnist_bags_sweep.sh` |
| Method validation on MNIST: density and activation function | `Default`, parent runs `density_check` | `main_mlflow.py` |
| AbMIL on GWHD: splits, activation functions, backbones, reduced density | `gwhd_attmil`, `gwdh_attmil` (gated attention and backbone controls), `cluster_test`, `test_reduced_pos` | `main_mlflow.py` |
| AbMIL on GWHD: single-parameter tests and greedy search | `gwhd_model_parameter_sweep`, `greedy_hpo`, `greedy_sigmoid` | `start_runs.sh`, `run_greedy.sh` |
| AbMIL on GWHD: colour and learnt grayscale | `color_models` (first comparison: `attention_gwhd_optimization_color`, `first_results`) | `run_color.sh` |
| Supervised baseline | not logged to MLflow | `models/patch_classifier.py` |
| CLAM-based model | `clam_test`, `clam_sweep`, `clam_sweep_v2`, `clam_sweep_v3`, `clam_instscores` | `run_cluster_clam_sweep.sh`, `run_cluster_clam_instscores.sh` |
| Pretrained backbone | `pretrained_test` | `main_mlflow.py --pretrained_backbone` |
| Transfer to the drone datasets | `synth_models`, `drone_test`, `clam_greedy_drone` | `run_synth.sh`, `run_greedy_clam.sh` |
| From patch to wheat ear counts | `pretrained_test` (`*_insteval`), `full_image_evaluation`, `full_image_evaluation_cal_threshold` | `run_full_image_clam_eval.sh` |

The other experiments in the database (e.g. `fpn_mil_gwhd`, `compare_datasets`,
`mnist_attention_dataset_optimization`) are preliminary tests that are not reported.

The `run_*.sh` scripts are SLURM jobs for the GPU cluster of the institute. They contain
the cluster user, paths and container image in a configuration block at the top that has
to be adapted. Results were copied back with `sync_mlflow.sh` and merged into
`runs/mlflow.db` with `import_mlflow.py`.

### Tables and figures

```bash
# browse the results
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db

# re-create the figures and tables of the thesis in eval/results/
bash eval/make_thesis_figures.sh

# own tables and figures from any experiment
python eval/mlflow_toolbox.py --help
```

`make_thesis_figures.sh` reproduces the generated figures of the thesis byte for byte
(PNG). Not covered by it: the feature visualisations (MLflow artifacts of the runs with
`--visualize_features`), the GWHD example figures in `eval/figures/` (the image crop was
chosen visually from a contact sheet, see the docstrings), and some tables that were
assembled by hand from MLflow queries.

### Tests

```bash
python -m pytest testing/
```

## Notes and known limitations

- **Centred GWHD patches (`--dense`).** `create_gwhd_bags.py` parses `--dense` but does
  not pass it on to `generate_patch_definitions`, so the centred patch definition is not
  reachable from the command line. The centred datasets (`gwhd_bags_dense*`) on the USB
  stick are the ones used in the thesis.
- **Reduced density.** No script in this repository creates `gwhd_reduced_pos.h5`.
- **Earlier GWHD file.** The experiments from May 2026 (`attention_gwhd_optimization*`,
  `first_results`, `sigmoid_v_softmax`, `gwhd_model_parameter_sweep`) read
  `bags/gwhd_bags.h5` before it was replaced on 2026-06-06. That earlier version is
  provided as `gwhd_bags_2026-05-19.h5`; its groups `gwhd_bags` and `gwhd_bags_dense`
  match the current file (spot check of 30 bags each), and it additionally contains the
  colour groups.
- **MNIST validation split.** Validation and test bags are drawn from the MNIST test set
  with the same seed and are therefore identical. The MNIST experiments use AbMIL, which
  tests the model after the last epoch, so the validation split is only logged there.
- **Overlap.** `compute_overlap` in `create_gwhd_bags.py` returns the fraction of the patch
  covered by a bounding box (intersection / patch area).

## Credits and license

Based on the AbMIL implementation by Maximilian Ilse and Jakub M. Tomczak (MIT license,
see `LICENSE`). The CLAM-based model follows
[mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM); deliberate deviations are listed
in `models/clam_model.py`.

- Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance
  Learning. ICML. [arXiv:1802.04712](https://arxiv.org/abs/1802.04712)
- Lu, M. Y., Williamson, D. F. K., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F.
  (2021). Data-efficient and weakly supervised computational pathology on whole-slide
  images. Nature Biomedical Engineering, 5, 555–570.
- David, E. et al. (2021). Global Wheat Head Detection 2021: An improved dataset for
  benchmarking wheat head detection methods. Plant Phenomics.
