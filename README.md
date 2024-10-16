# ANIE

Pytorch implementation of the Attentional Neural Integral Equations (ANIE): 
arXiv:2209.15190
https://arxiv.org/abs/2209.15190.


# Quickstart
Clone this repository locally:

```
git clone https://github.com/emazap7/ANIE.git
```


Create an Anaconda environment from the `environment.yml` file using:

```
conda env create --file environment.yml
conda activate anie
```


Citation [Zappala et al](https://arxiv.org/abs/2209.15190):
```
@article{zappala2022neural,
  title={Neural integral equations},
  author={Zappala, Emanuele and Fonseca, Antonio Henrique de Oliveira and Caro, Josue Ortega and van Dijk, David},
  journal={arXiv preprint arXiv:2209.15190},
  year={2022}
}
```

## Structure of the code
The main code is all located in the folder 'IE_source'. The file 'Attentional_IE_solver.py' contains the integral equation solver that uses attention for quadrature of the integrals. The file 'Galerkin_transformer.py' contains the transformer without softamx, which we modified from https://github.com/scaomath/galerkin-transformer, and it is called inside the solver for the quadrature/cubature. The file 'experiments' contains all codes for the experiments in the article. The functions to run the experiments are called in the jupyter notebooks. The file 'integrators' contains helper functions to perform integration (regular) for the integral equation solver that does not use attention. The file 'kernels' contains useful functions to produce integral equation datasets. The file 'solver.py' contains the implementation of the integral equation solver without attention (so, regular integration). In 'utils.py' some helper functions such as for saving the models, loading them etc., are found.  

The folder DeepONet+UNet contains our implementation of one of the models used for comparison, as described in the article. 

# Datasets

## Toy data 
The toy data has been obtained by solving analytical IDEs in 2D and 4Ds. The kernels
used in both cases were convolutional kernels where the entries were given by combinations of trigonometric functions. The F function was a hyperbolic cosine. To obtain the datasets, we have randomly sampled initial conditions and solved the corresponding initial value problem for the IDEs, using our implementation of the IDE solver. The integrals have been performed with Monte-Carlo integration with 1K sampled points per interval, and the number of iterations used was set to 10, which was empirically seen to guarantee convergence to the solution.

The scripts for the toy data generation can be found [here](https://arxiv.org/abs/2209.15190).


# Manual Environment Creation
If the `environment.yml` file does not successfully recreate the environment for you, you can follow the below steps to install the major packages needed for this project:

1. Create and activate an anaconda environment with Python version 3.10:
```
conda create -n anie python=3.10
conda activate anie
```

2. Install Pytorch: `conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch`

3. Install Matplotlib: `conda install -c conda-forge matplotlib`

4. Install `pip install git+https://github.com/patrick-kidger/torchcubicspline.git`

5. Install `conda install -c anaconda scipy`

6. Install `conda install -c conda-forge torchdiffeq`

7. Install `conda install -c conda-forge tqdm`

8. Install `conda install -c anaconda scikit-learn`



