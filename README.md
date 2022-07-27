# DynamICE (Dynamic IDP Creator with Experimental restraints)
A generative-reinforcement model (RL-GRNN) to generate new IDP conformer ensembles biased towards experimental data.

## Required packages
* Python (3.7.1)
* numpy 
* yaml 
* pytorch 
* idpconfgen (https://github.com/Oufan75/IDPConformerGenerator.git, modified from https://github.com/julie-forman-kay-lab/IDPConformerGenerator.git)
* sidechainnet (included, modified from https://github.com/jonathanking/sidechainnet.git)
* X-EISD (https://github.com/Oufan75/X-EISD.git, modified from https://github.com/THGLab/X-EISD.git)
### optional packages for plotting and analysis
* Biopython
* matplotlib
* seaborn
* numpy
* mdtraj
* scipy

## Training
Modify parameters in training files based on request.
* train_grnn.py for pretraining generative model.
* generate_rnn.py for generating conformers with trained models.
* run_rl.py for reinforcement learning with pretrained models.
