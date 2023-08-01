# Tractable Function-Space Variational Inference in Bayesian Neural Networks (FSVI)

This repository contains the official implementation for

**_Tractable Function-Space Variational Inference in Bayesian Neural Networks_**; Tim G. J. Rudner, Zonghao Chen, Yee Whye Teh, Yarin Gal. **NeurIPS 2022**.

**Abstract:** Reliable predictive uncertainty estimation plays an important role in enabling the deployment of neural networks to safety-critical settings. A popular approach for estimating the predictive uncertainty of neural networks is to define a prior distribution over the network parameters, infer an approximate posterior distribution, and use it to make stochastic predictions. However, explicit inference over neural network parameters makes it difficult to incorporate meaningful prior information about the data-generating process into the model. In this paper, we pursue an alternative approach. Recognizing that the primary object of interest in most settings is the distribution over functions induced by the posterior distribution over neural network parameters, we frame Bayesian inference in neural networks explicitly as inferring a posterior distribution over functions and propose a scalable function-space variational inference method that allows incorporating prior information and results in reliable predictive uncertainty estimates. We show that the proposed method leads to state-of-the-art uncertainty estimation and predictive performance on a range of prediction tasks and demonstrate that it performs well on a challenging safety-critical medical diagnosis task in which reliable uncertainty estimation is essential.

<p align="center">
  &#151; <a href="https://timrudner.com/fsvi"><b>View Paper</b></a> &#151;
</p>


## Environment Setup

First, set up the conda environment using the conda environment `.yml` files in the repository root, using

```
conda env create -f environment.yml
```

### Installing JAX

To install `jax` and `jaxlib`, use
```
pip install "jax[cuda11_cudnn86]==0.4.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Installing PyTorch (CPU)

To install `pytorch` and `torchvision`, use

```
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
```

NB: We recommend installing PyTorch for CPU-only to make sure that PyTorch does not interfere with JAX's memory allocation.


## Running Experiments

NB: Replace `path/to/repo` below by the absolute path to this repository.

**FMNIST**

To run FSVI with a small CNN on FashionMNIST, execute

```
python trainer_nn.py --config configs/fsvi-cnn-fmnist.json --config_id 0 --cwd path/to/repo
```

**CIFAR-10**

To run FSVI with a ResNet-18 on CIFAR-10, execute

```
python trainer_nn.py --config configs/fsvi-resent18-cifar10.json --config_id 0 --cwd path/to/repo
```

NB: The configs above do not use additional datasets to construct the context set. Instead, they use a corrupted (i.e., augmented) training set. To use another dataset (e.g., KMNIST or CIFAR-100), change the `--context_points` arg in the config. To to do this, simply change the `config_id` arg as shown below.

**FMNIST** (with KMNIST as context set)
```
python trainer_nn.py --config configs/fsvi-cnn-fmnist.json --config_id 1 --cwd path/to/repo
```

**CIFAR-10** (with CIFAR-100 as context set)
```
python trainer_nn.py --config configs/fsvi-resent18-cifar10.json --config_id 1 --cwd path/to/repo
```


## CIFAR-10 Corrupted Evaluation

By default, no CIFAR-10 Corrupted datasets are loaded. To evaluate model performance on these datasets, add `--full_eval` to the configs above and make the following manual modification to the `timm` library:

To load CIFAR-10 Corrupted configurations from TFDS, the following path is necessary in the `timm` library [parser_factory.py](https://github.com/rwightman/pytorch-image-models/blob/v0.6.7/timm/data/parsers/parser_factory.py#L9):

```diff
- name = name.split('/', 2)
+ name = name.split('/', 1)
```

if you do not make this change and use `full_eval` in the config, you will get an error stating

```
No builder could be found in the directory: ./data/CIFAR10 for the builder: speckle_noise_1.
```


## Out-of-Memory Errors

If you encounter an out-of-memory error, you may have to adjust the amount of pre-allocated memory used by jax. This can be done, for example, by setting

```
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
```

Note that this is only one of many reasons why you may encounter an OOM error.
