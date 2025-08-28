# Sentiment Classifier from Scratch
This project builds a sentiment classifier for IMDb movie reviews using a Kaggle dataset. I implemented the core algorithm manually with NumPy, deriving calculations with my calculus skills, following the "learn the hard way" approach inspired by Theodore Bendixson's video and Andrew Ng's ML course on YouTube.

## Problem Statement
Predict whether a movie review is positive or negative using the IMDb dataset from Kaggle, targeting an accuracy of 75%+ with a scratch-built model.

## Approach
- Preprocessing: Tokenized text and normalized features by hand using basic Python and NumPy.
- Training: Implemented a simplified linear regression-based classifier manually, deriving the cost function and gradient updates with calculus.
- Challenge: Initially struggled with text data representation; resolved by creating a basic bag-of-words model and debugging numerical overflow manually.

## Results
***Initial test (e.g., Aug 28, 2025): [Insert accuracy, e.g., 70%] on a small subset. Plan to refine with more data. Update with final accuracy and analysis after completion.***

## Code Structure
- `/data/`: IMDb CSV files or raw text data downloaded from Kaggle.
- `/scripts/classifier.py`: Main Python file with manual implementation of the classifier.
- `/notebooks/experiments.ipynb`: Jupyter notebook for experiments, derivations, and initial tests.

## Math Derivations
### Linear Regression Cost Function
The cost function for linear regression is defined as:
\( J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 \)
where \( h_\theta(x) = \theta^T x \) is the hypothesis, \( m \) is the number of examples, and \( \theta \) are the parameters.

### Gradient Descent Update
Derived the gradient update rule manually:
\( \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \)
Adjusted the learning rate \( \alpha \) by hand to stabilize convergence, using my calculus skills to minimize \( J(\theta) \).

## Lessons Learned
***Update with reflections after completion, e.g., "Learned to handle text data manually, improved gradient descent tuning. Next Steps: Add cross-validation, compare with scikit-learn."***

## Acknowledgments
Inspired by Andrew Ng's ML course on YouTube (Video 1: Linear Regression). Dataset sourced from Kaggle.
