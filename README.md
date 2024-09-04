# Exploring Domain Adaptation for Floor Plan Vectorization

*Jeroen Hofland*

This research explores the challenges of converting architectural floor plans from raster to vector images. Unlike previous studies, our research focuses on domain adaptation to address stylistic and technical variations across different floor plan datasets. We develop and test our vectorization method on the CubiCasa5K benchmark, which includes 3 different floor plan styles. Our analysis reveals differences in input features across the CubiCasa5K styles, indicating the potential for domain adaptation research, mostly in room segmentation. However, we also find multiple indications that labelling in the CubiCasa5K dataset is ambiguous and inconsistent. Furthermore, styles with more training data do not always perform better, highlighting the complexity differences between floor plan styles. Our baseline shows a 0.7\% gap for rooms yet a 0.6\% improvement for objects, likely caused by the smaller feature gaps and inconsistent labelling. To address the adaptation gap, we add a Multi-Kernel Maximum Mean Discrepancy (MK-MMD) loss to the CubiCasa5K model to minimize feature distribution differences between domains. While our MK-MMD implementation shows potential for reducing the adaptation gap, persistence issues and mixed results across classes make it difficult to draw clear conclusions. Our findings also show the role of balancing spatial context in the MK-MMD calculation. These insights lay a foundation for future domain adaptation research in floor plan vectorization.

## Installation

CUDA: `conda env create -f environment-cuda11.3.yml`

CPU: `conda env create -f environment.yml`

## Usage

Use command-line arguments to override the defaults given in `config.yaml`. For example:

```bash
python train.py dataset.name=cubicasa
```

## Data experiments

The data experiments can be found in the `paper` folder. We also added two experiments in the main folder called `exp_ambiguity_check.py` and `exp_domain_performance.py`. Archived experiments can be found in the `exp_achive` folder.

**HPC**: to run on the HPC/DAIC, copy your code to the HPC/DAIC, adapt the given `run_cluster.sbatch` to your HPC/DAIC settings (see the top of the file) and use it by appending the Python call to the call to the sbatch file:

```bash
sbatch run_cluster.sbatch
```

### Adding models

- Add the code for the model in a new file in `models`;
- Import & call the new model in `model_factory.py`

### Adding datasets

- Add the code for the dataset in a new file in `datasets`. Make sure to make methods for creating dataloaders for train and val/test.
- Import & call the new methods in `dataset_factory.py`

### Resuming training from a checkpoint

W&B saves checkpoints as "artifacts".

- Use code like below to make W&B download the `Runner` checkpoint to disk:

```python
artifact_name = f"{cfg.wandb.entity}/{project_name}/{artifact_name}"
print(artifact_name)
artifact = wandb_logger.experiment.use_artifact(artifact_name)
directory = artifact.download()
filename = os.path.join(directory, 'model.ckpt')
```

- Add flag `ckpt_path=filename` to the call to `Trainer.fit()`
- Consider generalizing this by making `artifact_name` given by a new config key `cfg.resume.artifact`
