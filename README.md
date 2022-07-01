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
![image](https://user-images.githubusercontent.com/11977931/176827092-15a0ae49-4042-4840-84bc-7b1d97ff38d0.png)

## Classification Report
```
==================== Train ====================
              precision    recall  f1-score   support

           0       0.46      0.83      0.59     39990
           1       0.96      0.81      0.88    213466

    accuracy                           0.82    253456
   macro avg       0.71      0.82      0.74    253456
weighted avg       0.88      0.82      0.84    253456

==================== Test ====================
              precision    recall  f1-score   support

           0       0.44      0.81      0.57      9959
           1       0.96      0.81      0.88     53406

    accuracy                           0.81     63365
   macro avg       0.70      0.81      0.73     63365
weighted avg       0.88      0.81      0.83     63365
```

`Precision` should be what we focus on and it's 0.96 on test set.

Since our positive label is `Fully Paid`, that means from the predicted positive borrowers (TP+FP), how many are classified as Fully Paid correctly (TP).

This should also align with the business operation in a way that, if they 'trust' the model, and approved the loan requests for `100` customers, `96` customers are correctly classified as Fully Paid.

## Score distribution (Normalized)
![image](https://user-images.githubusercontent.com/11977931/176827449-e1fc31d6-5905-4b3e-a0e9-8e5fc322a55a.png)

## Concern on using ZIP feature
`ZIP` code is what I concern with its validity and it quite dominates the result.  
Not using zip code, I got ROC-AUC around `0.72`. By putting it in, it goes up to `0.90`  
In real practice, we may need to consider how this address is collected. Is it the contact address, home address? It can be changed. How old is this data. There are things to consider putting this into use.
![image](https://user-images.githubusercontent.com/11977931/176828474-29f56651-1eb8-4919-84d6-032f4024e327.png)



# Model Insights

## SHAP Beeswarm
This tells a lot.
![image](https://user-images.githubusercontent.com/11977931/176827563-bfa8eb7e-cc7c-4a29-96fb-61affd95f0bf.png)
- `zip`  
Apart from my concern, it seems like some zip codes impact highly impact on both negative and positive ways.
- `sub_grade`  
The more grade gives negative impact in less probablity of Fully Paid (To be precise, log odd not probability). This seems true as G as encoded higher than A. The lower subgrade boosts chance of Fully Paid
- `installment_inc_ratio`
This ratio kinda represents the future debt obligation if the loan got accepted. The more of this value, the less of being Fully Paid

## Mean SHAP Plot
If the beeswarm above is too complicated for explaining to business users and stakeholders. Here goes this plot, just to show its impactness without direction.
![image](https://user-images.githubusercontent.com/11977931/176827573-f224666a-8505-4b62-a97a-154a767eeff2.png)



