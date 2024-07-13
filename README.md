# EMLART-GP - Tutorial GECCO 2024

This is a repository for the Tutorial at GECCO 2024: [Evolutionary Art and Design in the Machine Learning Era](https://gecco-2024.sigevo.org/Tutorials#id_Evolutionary%20Art%20and%20Design%20in%20the%20Machine%20Learning%20Era) 


## Installation

The following commands will install the required packages to execute the code:

```bash
conda create --name emlart-gp-tutorial python=3.10
conda activate emlart-gp-tutorial 
conda install -c conda-forge tensorflow
conda install matplotlib scikit-image pytorch-lightning -c pytorch
python -m pip install git+https://github.com/openai/CLIP.git         
```
You will also need the weights for the models used by the provided scripts from the following link:

[models](https://www.dropbox.com/s/vusdr3oo5htfqh9/models.zip?dl=1) 

unzip the file ensuring that a folder called "models" is inside the cloned repository 
```bash
models/
tensorgp/
stablediffusion_examples/
...
```

## Usage

To execute the emlart-gp approach use the following command on the terminal in the repository folder:

```console
python emlart_gp.py <starting random seed number> <#of runs> <# of generations> <text prompt>
```

E.g.
```bash
python emlart_gp.py 10 1 30 "sunset, bright colors" 
``` 
