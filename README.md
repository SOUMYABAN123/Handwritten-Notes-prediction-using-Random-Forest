🧠 Task 2: Handwritten Digit Classification using Random Forest

Author: Soumya Sekhar Banerjee
Matriculation Number: 100004430
University: SRH University of Applied Science
Course: Masters in Engineering – Artificial Intelligence

📄 Abstract

Random Forest is a supervised ensemble learning algorithm used for both classification and regression.
It builds multiple decision trees during training and outputs the class that is the mode of the predictions (for classification) or the mean prediction (for regression).

It is based on the principle of “wisdom of the crowd” — combining multiple weak learners (decision trees) to form a strong overall model with reduced variance and improved generalization.

Key Characteristics

Ensemble method: Combines multiple decision trees.

Randomness: Uses random subsets of data and features.

Stability: Reduces variance compared to a single decision tree.

Interpretability: Provides feature importance estimates.

🎯 Aim of the Experiment

To build a supervised ensemble classifier for handwritten digits (0–9) using the Random Forest algorithm on the MNIST dataset.

Dataset: MNIST (CSV format with 784 features per sample)

Output: Predicted digit label (0–9)

Evaluation: Accuracy and confusion matrix

🧩 Mathematical Intuition
1. Ensemble of Decision Trees

Each tree is trained on a bagging sample (sampled with replacement).
At each split, a random subset of features is considered.

2. Decision Tree Function

Each decision tree recursively partitions the feature space using decision rules until a stopping condition is met (e.g., max depth).

3. Aggregation of Predictions

For classification:

𝑦
^
=
mode
{
ℎ
1
(
𝑥
)
,
ℎ
2
(
𝑥
)
,
.
.
.
,
ℎ
𝑛
(
𝑥
)
}
y
^
	​

=mode{h
1
	​

(x),h
2
	​

(x),...,h
n
	​

(x)}

Where 
ℎ
𝑖
(
𝑥
)
h
i
	​

(x) is the prediction of the i-th tree.

For regression:

𝑦
^
=
1
𝑛
∑
𝑖
=
1
𝑛
ℎ
𝑖
(
𝑥
)
y
^
	​

=
n
1
	​

i=1
∑
n
	​

h
i
	​

(x)
4. Bias–Variance Tradeoff

Combining multiple de-correlated trees reduces variance:

𝑉
𝑎
𝑟
(
𝑌
ˉ
)
=
𝜌
𝜎
2
+
(
1
−
𝜌
)
𝜎
2
𝑛
Var(
Y
ˉ
)=ρσ
2
+
n
(1−ρ)σ
2
	​


As the number of trees 
𝑛
→
∞
n→∞, variance decreases.

🔁 Task Overview and Workflow

Algorithm: Random Forest
Input: Pixel intensities (0–255) from MNIST images
Output: Predicted digit (0–9)

🧠 Training Steps

Load the data

Read MNIST CSV file (label + 784 pixel features).

Preprocessing

Remove NaN values, optional normalization.

Separate features and labels

X: pixel values, y: labels (0–9).

Initialize Random Forest

Define number of trees, max depth, and random seed.

Train the model

Each tree is trained on a bootstrap sample and random subset of features.

Out-of-Bag (OOB) validation

Estimate performance using samples not seen by each tree.

Prediction

Aggregate majority votes across trees.

Evaluation

Compute accuracy, confusion matrix, classification report.

Hyperparameter tuning

Test multiple tree counts: 10, 50, 100, 250, 500.

Select final model

Optimal trade-off between performance and runtime.

📊 Model Evaluation: Confusion Matrix
Digit	Interpretation Summary
0	968 correct, minor confusions with 2, 5, 6
1	1122 correct, excellent precision
2	984 correct, confusions with 3, 8, 7
3	956 correct, confusions with 5, 8, 9
5	841 correct, often mistaken for 3 or 8
8	905 correct, often confused with 3, 5, 9
Key Insights
Metric	Observation
Accuracy	95.59%
Most Accurate Digits	1, 0, 7
Most Confused Digits	3, 5, 8, 9
Model Generalization	Strong, minimal overfitting
🌲 Random Forest Parameters
1. n_estimators: Number of Trees
n_estimators	Effect	Typical Result
10–50	Too few trees → underfitting	Low accuracy
100–300	Reasonable ensemble size	High accuracy
500–1000	Stable, less variance	Slightly slower
>1000	Diminishing returns	Negligible gain

More trees improve averaging but increase runtime.

2. oob_score=True: Out-of-Bag Evaluation

Out-of-Bag (OOB) samples are those not used in tree training.
OOB accuracy serves as an internal validation metric — similar to cross-validation.

Parameter	Role	Interaction
n_estimators	Controls number of trees	More trees → more stable OOB
oob_score	Uses left-out data for validation	Needs enough trees for reliability

OOB ≈ Test accuracy → confirms model generalization.

⚙ Model Variants and Performance
Model	Key Settings	Description	Accuracy
rf1	100 trees	Baseline	~95%
rf2	50 trees	Fewer trees → underfit	↓
rf3	250 trees	More stable	↑
rf4	500 trees	Best trade-off	~95.6%
rf5–rf7	750–1000 trees	Diminishing returns	≈ same
rf8	500 trees, max_features=2	More randomness	≈ rf4
📈 Effect of n_estimators on Accuracy

Interpretation of Performance Curve

Observation	Explanation
At 10 trees	OOB ≈ 0.84, Test ≈ 0.92 → underfitting
At 50 trees	Sharp rise in accuracy
At 100 trees	Stable accuracy ~95.5%
At 250–500 trees	Plateau — best balance
Beyond 500	No real improvement

Conclusion:

The optimal configuration lies between 250–500 trees, giving accuracy ≈ 95.6–96% with minimal variance.

⚙️ Hyperparameter Tuning

Key hyperparameters:

n_estimators

max_features

max_depth

min_samples_split

min_samples_leaf

bootstrap

oob_score

Manual tuning focused on:

Increasing n_estimators

Adjusting max_features

This improved accuracy from ~92% → ~96%.

🧾 Conclusion

The Random Forest classifier achieved ~95.6–96% accuracy on MNIST.

Best performance with:

n_estimators ≈ 500

max_features = sqrt(features) or 2

OOB ≈ Test accuracy → strong generalization.

Most misclassifications were between visually similar digits (3–5–8–9).

Random Forest proved to be a robust, interpretable, and high-performing baseline for handwritten digit recognition.

📚 References

BuiltIn: Random Forest Algorithm

Decision Trees and Random Forests (Gnjatovic)

Random Forest in R (Gnjatovic)

Google Research Papers

Udemy & Copilot Resources
