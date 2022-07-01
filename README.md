# Digital Ventures Interview
By: Tanat Iempreedee

As a part of job interview with Digital Ventures, please find my deliverables in this repository.
My main tool to develop the model is Python using Jupyter Notebook.

## Objective
Build a classfication model to score the probability of a borrower having `Fully Paid` loan status.
- Task: Binary Classification
- Labeling: Positive (1) - Fully Paid, Negative (0) - Charged Off
- Unit of analysis: Borrower
- Address imbalanced problem: pos/neg : 84/16
- Algorithm: XGBoost

I decdied on using XGBoost only and not to try other models due to time limitation. And also not using cross-validation, it will take long time to process.

## Code Files
In order to keep the code not too long, I divided them into 3 parts
- `00 Clean.ipynb` is for data cleaning because the given dataset is not well-formed csv.
- `01 Profiling.ipynb` is for basic EDA, using Pandas Profiling which can't be seen if you open from github directly.
- `02 EDA.ipynb` explore data in terms of distributions between 0 and 1 classes.
- `03 Modeling.ipynb` the code for preprocessing and build the model.


# 0) Data Cleaning
`00 Clean.ipynb`
- `pandas.read_csv` doesn't work with some corrupted records. Using `csv.reader` instead
- Identify the corrupted rows by simply check the number of parsed tokens. If it's not `27`, remove.  
- Removing problematic rows, and the whole data can be converted into `DataFrame`

# 1) Data Profiling
`01 Profiling.ipynb`
Using pandas profiling, run an analysis on the original dataset. Then I tried preprocess and generate some more features and run again to get the idea on how the processed and generated features look like.

# 2) EDA
`02 EDA.ipynb`
Explore each of the features manually, seeing distribution, descriptive statistics between 2 groups of Charged Off and Fully Paid.

# 3) Modeling

## Feature engineering
Some generated features
- `installment_inc_ratio`  
$$ \frac{MonthlyInstallment}{AnnualIncome/12} $$
Similar to DTI (debt-to-income), but according to the data dict, the requested loan amount is not included in the calculation. So I made this to measure the additional debt obligation to the borrower if the loan is accepted.
- `mort_acc_missing`  
Since 10% of the whole dataset of `mort_acc` (the number of mortgage accounts) are missing. So I imputed zero for missing values, and create a variable to tell that `mort_acc` is missing
- `inc_not_verified`  
Whether or not the reported income is verified based on `verification_status`. Not Verified goes to 1, else 0.
- `zip`  
Zip code extracted from `address` which I still have doubt the correctness since it has such small cardinality for zip code. It turns out to be the feature with the most predictive power.

## Categorical encoding
Basically we deal with nominal category with One-hot encoding. But after some experiments, I found that One-hot and Oridnal are not significantly different for this dataset.  

At the end, I use `Ordial Encoding` for all categorical variables. The tree-based models are capable to learn from label-encoded nominal categories.

## Holdout test set
Load preprocessed dataset from earlier step and split it into `Train`, `Validation` and `Test` sets with 0.6/0.2/0.2 proportion respectively.

## Feature selection
- Check `Mutual Information Scores`  
To see predictive power of features with Mutual Information scores. 
- Try fitting a model using default parameters and see the feature importances in consideration of choose features

## Model Training
- Use `XGBoost` only (The original idea was to use RandomForst as a baseline, and try LightGBM, and other models)
- One way to address imbalanced problem using `XGBoost` is to specify `scale_pos_weight` parameter like this  
```Python
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
```
- `roc-auc` as the evaluation metrics.
- Try resampling methods - `Under-sampling`, `Over-sampling` and `SMOTE`. None of them improves the performance. `SMOTE` tends to give overfitting result.
- Hand-tune hyperparameter, the performance improves a bit.
- Final Model - use Train+Validation to train and it improves Test Set performance.  
I did make sure that I didn't use Test set in the `eval_set` parameter to make sure that test set is untouched

| Experiment | Train-AUC | Valid-AUC | Test-AUC | Note |
|---|---|---|---|---|
| Default params | 0.9171 | 0.9066 | 0.9090 | |
| Default params + Under-sampling | 0.9233 | 0.9061 | 0.9074 | |
| Default params + Over-sampling | 0.9181 | 0.9060 | 0.9083 | |
| Default params + SMOTE | 0.9159 | 0.8607 | 0.8644 | Overfitting |
| Tuned (No resampling) | 0.9170 | 0.9088 | 0.9102 | |
| Final Model | 0.9199 | N/A | 0.9106 | Best |

# Performance
## ROC Curve
![image](https://user-images.githubusercontent.com/11977931/174436934-22457ce7-40df-4fab-b473-46ec0f93e903.png)
## Classification Report
![image](https://user-images.githubusercontent.com/11977931/174437404-451a5510-a1cf-415b-9556-4591f9269aaf.png)

`Precision` should be what we focus on and it's 0.91 on test set.

Since our positive label is 'Fully Paid', that means from the predicted positive borrowers (TP+FP), how many are classified as Fully Paid correctly (TP).

This should also align with the business operation in a way that, if they 'trust' the model, and approved the loan requests for 100 customers, 91 customers are correctly classified as Fully Paid.

# Model Insights
- `Feature Importance`
- `Permuation Importance`
- `SHAP`

All 3 reports are somehow aligned in the same way, but I'll go through the insights with `SHAP`.
## SHAP Summary Plot
![image](https://user-images.githubusercontent.com/11977931/174437707-e18a608d-5b61-489f-95f4-c7c64362d028.png)

- `sub_grade` and `grade` are encoded in such order that G > F > ... > B > A. So A is on low value, and G is on high. The plot says that the high values impact the probability in a negative way. 
- `term` says `60 months` has negative effect and `36 months` has positive effect
- `mort_acc_missing` is interesting and weird. If `mort_acc` is missing or not provided, it raises the probability of Fully Paid.
- Generally the features that impact the prob. negatively (high value -> decreases prob.), for example,
`revol_util`, `dti`, `open_acc`, `installment_inc_ratio`, `loan_amnt`
- And the features that impact the prob. positively (high value -> increases prob.), such as, `annual_inc` which is somehow a common sense.

## Mean SHAP Plot
The `Summary Plot` seems to pack a lot of information, but it may be difficult to interpret for non-tech or business users.  
Need a simple way to explain? Use Mean SHAP instead, easy to interpret, it simply tells how much pact each feature has to the prob. of being Fully Paid.
But we lost the impact direction. 

![image](https://user-images.githubusercontent.com/11977931/174438419-1d54f4db-0354-4291-bfa9-197da4efc8e4.png)

# How to test deliverables
```Python
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv('data/test_final.csv')
y_test = X_test.pop('loan_status')

y_score = model.predict_proba(X_test)[:,1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_score)}")
```

### Remark
The code is not so productionization-friendly. Should use `Pipeline`. In the real work environment, this would be redesigned to make a proper preprocessing pipeline for model serving.


