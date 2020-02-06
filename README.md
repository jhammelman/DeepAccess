# DeepAccess

Code for training, testing, and motif extraction from ensemble of convolutional neural networks for multilabel classification from DNA sequence.

## Dependencies

All dependencies can be installed from conda using the provided environment file:

`conda env create -f ensembleNN.yml`

Make sure to activate environment prior to running code.

Note this conda enviroment is for running models on GPU. Code and conda environments will need to be modified if you would like to run these models on CPU.

## Training
Training takes in a fasta file of DNA sequences, a file with labels for each seqeuence, and an output folder where the trained ensemble will be stored.

Example:
`python train_ensemble.py data/train.fa data/train_act.txt example/`

## Testing
Testing takes in a fasta file, a folder where the model is stored, and the name of the outfile for model predictions.

Example:
`python test_ensemble.py data/test.fa example/model_predictions.txt`

## Extraction
Sequence saliency takes in a fasta file, a comparisons file for discriminative class comparisons, the prefix to store the importance for each individual model, a folder where the model is stored, and file prefix to store the model importance

Example:
`python extract_importance_ensemble.py data/test.fa data/comparisons.txt example example/test_saliency`