# Learning identity effects with neural networks

This code can be used to reproduce the numerical experiments in the paper

Invariance, encodings, and generalization: learning identity effects with neural networks
by S. Brugiapaglia, M. Liu, P. Tupper
https://arxiv.org/abs/2101.08386

## Setup and Installation

The experiments were run on the deep learning platform [Floydhub](https://www.floydhub.com/). 
You can find the list of environments supported by Floydhub as well as the Docker images at [https://docs.floydhub.com/guides/environments/](https://docs.floydhub.com/guides/environments/).
For this project, the `TensorFlow 1.5.0 + Keras 2.1.6 on Python 3.6` environment was used.

The Nvidia setup is as follows:
| Machine | Nvidia CUDA  | Nvidia CuDNN |
|---------|--------------|--------------|
| GPU     | Cuda v9.1.85 | CuDNN 7.1.2  |
| GPU2    | Cuda v9.1.85 | CuDNN 7.1.2  |


Install dependencies via `pip install -r requirements.txt`

## IE Model

`main.py` runs the identity effects experiments. The available command line arguments are:
```
Experiment Options

positional arguments:
  {FFWD,LSTM}           Type of IE model (FFWD or LSTM)
  {ALPHA,MNIST}         Data type of the experiment (ALPHA or MNIST)

optional arguments:
  -h, --help            show this help message and exit
  -i ITERATIONS, --iterations ITERATIONS
                        Number of epochs for each run
  -u UNITS, --units UNITS
                        Number of hidden units for each model
  -e {one-hot,distributed,haar}, --encoding {one-hot,distributed,haar}
                        Type of encoding for ALPHA experiments. Only set if data is ALPHA
  -c CV_OUTPUT, --cv-output CV_OUTPUT
                        Filepath for CV model output file.
  -r RUNS, --runs RUNS  Number of runs per experiment
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -o {ADAM,SGD}, --optimizer {ADAM,SGD}
                        Optimizer for the learning algorithm
  -f OUT_FOLDER, --out-folder OUT_FOLDER
                        Name of output folder
  -v, --verbose         Output verbosity
```

For example, the command `python main.py FFWD ALPHA -i 5000 -u 256 -e distributed -r 40 -l 0.025 -o SGD -v` runs the FFWD Alphabet Distributed experiment.

You can also run the experiments from `experiments.ipynb`.

## CV Model

The CV model outputs are saved in `CV_output`. If you want to retrain the CV model to produce new outputs, you can use the `train_CV.py` file. 
The available command line arguments are:

```
CV Model Training Options

optional arguments:
  -h, --help     show this help message and exit
  -f OUT_FOLDER  Output folder of saved training weights
  -e EPOCHS      Number of epochs to train
  -v, --verbose  Output verbosity
```

You can run `python train_CV.py -f <out_folder> -e 100 -v` to retrain the CV model for 100 epochs. The weights and loss/accuracy data will be saved in `<out_folder>`.

From there, produce CV model outputs using `predict_CV.py`. The available command line arguments are:

```
CV model Prediction Options

optional arguments:
  -h, --help       show this help message and exit
  -w WEIGHTS_FILE  Weights file
  -d {TEST,TRAIN}  Predict on MNIST test or MNIST train set
  -f OUT_FILE      Output filpath for CV predictions
```

For example, run `python predict_CV.py -w <weights_filepath> -d TEST -f <out_folder>` to produce CV model outputs using weights saved in `<weights_filepath>` on the MNIST test dataset.

Finally, rerun the IE experiments with `main.py`, this time using the `--cv-output` or `-c` argument to specify the location of the CV model output csv file (you can omit the `-e` flag).

Again, this can also be done from `CV_model.ipynb`.

## Results

View results and graphs from `results.ipynb`.
