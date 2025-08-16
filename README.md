# Rice Variety Classification with PyTorch

This project demonstrates how to classify rice with PyTorch. The workflow includes data preprocessing, model building, training, evaluation, and inference.

## Features

- **Dataset Loading:** Loads and preprocesses rice data from a csv file.
- **Model Architecture:** Implements a custom neural network for classification.
- **Training Loop:** Trains the model with validation and tracks accuracy.
- **Evaluation:** Tests the model on unseen data and reports performance metrics.
- **Inference:** Predicts rice variety from new samples.

## Getting Started

1. **Clone the repository and install dependencies:**
    ```bash
    git clone <repo-url>
    cd Rice-Classification-Pytorch
    pip install -r requirements.txt
    ```

2. **Run the notebook:**
    Open `classify_rice.ipynb` in Jupyter Notebook or VS Code and follow the steps.

## Directory Structure

```
.
├── classify_rice.ipynb
├── README.md
└── rice_Classification.csv
└── .gitignore
```

## Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- matplotlib
- torchinfo

## Results

The model achieves high accuracy in classifying rice varieties. See the notebook for detailed results and visualizations.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
