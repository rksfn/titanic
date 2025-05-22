# Titanic Survival Prediction using Neural Networks

This project implements a neural network model to predict survival on the Titanic dataset. The model uses various features from the passenger data to predict whether a passenger survived the Titanic disaster.

## Project Structure

```
titanic/
├── titanic_nn.py          # Main neural network implementation
├── requirements.txt       # Project dependencies
├── train.csv             # Training data
├── test.csv              # Test data
├── submission.csv        # Generated predictions
├── best_model.h5         # Saved model
├── training_history.png  # Training metrics visualization
└── confusion_matrix.png  # Model evaluation visualization
```

## Features Used

- Pclass (Passenger Class)
- Sex
- Age
- SibSp (Number of Siblings/Spouses)
- Parch (Number of Parents/Children)
- Fare
- Embarked (Port of Embarkation)

## Model Architecture

The neural network consists of:
- Input layer (7 features)
- Hidden layer 1: 64 neurons with ReLU activation
- Dropout layer (0.2)
- Hidden layer 2: 32 neurons with ReLU activation
- Dropout layer (0.2)
- Hidden layer 3: 16 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/rksfn/titanic.git
cd titanic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python titanic_nn.py
```

## Model Performance

The model's performance is evaluated using:
- Validation accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix

Training history and confusion matrix visualizations are saved as PNG files.

## Kaggle Submission

To submit predictions to Kaggle:
```bash
kaggle competitions submit -c titanic -f submission.csv -m "Neural Network with feature engineering"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 