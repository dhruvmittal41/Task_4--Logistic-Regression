# Task-4-Logistic-Regression.
## Tools Used

- **Python**
- **Pandas** – data handling
- **Scikit-learn** – model training and evaluation
- **Matplotlib** – visualization

---

## Steps to Run the Project

1. **Load Dataset**  
   Use any binary classification dataset (e.g., breast cancer dataset from scikit-learn).

2. **Preprocess the Data**  
   - Split into training and testing sets  
   - Standardize features using `StandardScaler`

3. **Train the Model**  
   Use `LogisticRegression` from `sklearn.linear_model`.

4. **Evaluate the Model**  
   - Confusion Matrix  
   - Precision and Recall  
   - ROC-AUC Score  
   - ROC Curve Plot

5. **Tune Threshold**  
   Adjust the default threshold (0.5) to improve precision or recall based on the problem's need.

6. **Understand Sigmoid Function**  
   Logistic regression uses the sigmoid function to convert outputs into probabilities:
   \[
   \sigma(z) = \frac{1}{1 + e^{-z}}
   \]
   Where `z` is the weighted sum of inputs.

---

## Metrics Explained

- **Confusion Matrix**: Shows TP, FP, FN, TN
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **ROC-AUC**: How well the model distinguishes between classes

---

## Sample Threshold Tuning

```python
# Adjust threshold from default 0.5
threshold = 0.3
y_pred_custom = (model.predict_proba(X_test_scaled)[:, 1] > threshold).astype(int)
