# IDC8208 Pedestrian Situated Intent Project

This project contains the codebase for the PSI Task 1 - Pedestrian Intent. The project showcases multiple iterations with changes to the model after each iteration. They are as follow:

- Baseline code provided by IEEE ITSS PSI Competition Team
- Adding Convolutional Layer to the baseline LSTM model
- Fine-tuning the Convolutional Layer
- Switching to Transformer-based model
- Adding Class Weights to the Transformer model
- Adding Focal Loss to the previous iteration

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install -r requirements. txt
```

## Dataset

You may request access to the dataset from the [IUPUI-CSRC Pedestrian Dataset](http://pedestriandataset.situated-intent.net/) website.

Please refer to the official [PSI Competition Dataset](https://github.com/PSI-Intention2022/PSI-Dataset) github for detailed instructions on obtaining and preparing the dataset as well as the required python scripts to process the dataset. Please note that the file locations mentioned in their README occasionally does not tally with the file tree structure of the dataset.

The key steps in preparing the dataset are:
- Splitting the videos into frames
- Extending the Key-Frame Cognitive Annotations


## Getting Started

For more information on the baseline model, please refer to the official [PSI Intent Prediction](https://github.com/PSI-Intention2022/PSI-Intent-Prediction) github. It explains about the dataset structure and splits.

The parameters/arguments used can be adjusted in the `./PSI-Intent-Prediction/opts.py` file.

To run the program:

```bash
cd PSI-Intent-Prediction
python main.py
```

