# DynamICE (Dynamic IDP Creator with Experimental restraints)
A generative to generate new IDP conformer ensembles biased towards experimental data.

## Required packages
* Python (3.7.1)
* numpy 
* yaml 
* pytorch (1.4.0) 
* idpconfgen (https://github.com/Oufan75/IDPConformerGenerator.git/tree/grnn, modified from https://github.com/julie-forman-kay-lab/IDPConformerGenerator.git)
* sidechainnet (included, modified from https://github.com/jonathanking/sidechainnet.git)
* X-EISD (https://github.com/THGLab/X-EISD.git)

## Training
Modify parameters in training files based on request.
* train_grnn.py for pretraining generative model.
* generate_rnn.py for generating conformers with trained models.
* train_biased_grnn.py Biasing pretrained models with exp data.
* torsion_splits.py for extracting backbone and sidechain torsion angles for pre-training. 
