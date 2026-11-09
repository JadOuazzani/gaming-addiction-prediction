# Gaming Addiction Prediction

Predicting a person's level of video-game addiction from their gaming habits and mental-health data, with machine-learning models implemented in NumPy (no scikit-learn).

Course project for EPFL's CS-233 *Introduction to Machine Learning* (spring 2026), built by Jad Chahed Ouazzani, Riad Elhariry and Sara Iraqi.

## The problem

The data is a subset of the Gaming and Mental Health dataset: 13 features describing each person's gaming habits and mental health (1,600 training and 400 test samples). Two tasks use the same inputs:

- **Classification**: predict the addiction level, *Low*, *Medium* or *High*. The classes are strongly imbalanced (69% / 28% / 3%).
- **Regression**: predict a continuous addiction score between 0 and 10.

## Part 1: linear models and k-NN

- **Linear regression**, closed-form least squares
- **Multinomial logistic regression**, trained by batch gradient descent with a numerically stable softmax
- **k-nearest neighbours** for both tasks, with Euclidean (L2) or Manhattan (L1) distance

Hyperparameters were chosen on a held-out validation split, with standardisation computed on the training split only.

**Results on the test set**: the linear models clearly beat k-NN on both tasks. In regression, linear regression reaches an MSE of 0.994 against 1.832 for k-NN. In classification, the class imbalance makes macro-F1 the honest metric: k-NN reaches 79.75% accuracy but a macro-F1 of only 0.51, and finds 1 of the 13 *High* cases, while logistic regression finds 12. The full analysis is in [`part-1/report.pdf`](part-1/report.pdf).

## Part 2: neural network and clustering

- **Multi-layer perceptron** with configurable layers, Sigmoid and ReLU activations, several loss functions and weight decay, with forward pass and gradients written by hand
- **K-means** clustering

## Structure

```
part-1/   linear regression, logistic regression, k-NN, report
part-2/   MLP, activations, losses, K-means
```

Each part has a `main.py` (training and evaluation from the command line, for example `python main.py --method knn --task classification`) and the course's `test_ms*.py` structure tests. The dataset was provided by the course and is not included: place it at `data/features.npz` inside each part, or pass `--data_path`.

## Team

| Member | Part 1 | Part 2 |
|---|---|---|
| Jad Chahed Ouazzani | k-NN (classification and regression, L1/L2), logistic regression integration, report | — |
| Riad Elhariry | Project structure, linear regression | Activations, MLP, training logic |
| Sara Iraqi | Validation split and normalisation, softmax stability fix | Losses, K-means, data preparation |

The full commit history of both parts is preserved.
