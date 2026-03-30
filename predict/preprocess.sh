#!/usr/bin/env bash

# Dataset used in pre-training.
#dataset_trained=QM9under14atoms_atomizationenergy_eV
# dataset_trained=QM9full_atomizationenergy_eV
#dataset_trained=QM9full_homolumo_eV  # Two properties (homo and lumo).
# dataset_trained=yourdataset_property_unit
dataset_trained=QM9under7atoms_homolumo_eV

# Basis set and grid field used in pre-training.
basis_set=6-31G
radius=0.75
grid_interval=0.3

# Dataset for prediction.
dataset_predict=yourdataset_property_unit  # Extrapolation.

python preprocess.py $dataset_trained $basis_set $radius $grid_interval $dataset_predict
