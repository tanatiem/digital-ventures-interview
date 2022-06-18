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
- `01 Clean.ipynb` is for data cleaning.
- `02 Prep & EDA.ipynb` is for basic EDA, feature engineering, and data preprocessing
- `03 Modeling.ipynb` is the main part where I split datasets, train the model, report model performance and interpretability.

Please note that each part produce output files after it's done. This is my personal habit to save state of the work.  
In case something crashes, I don't have to restart all over. But the file are too big to upload due to GitHub limitation, not all the files are uploaded.

# 1) Data Cleaning
`01 Clean.ipynb`
- `pandas.read_csv` doesn't work with some corrupted records. Using `csv.reader` instead
- Identify the corrupted rows by simply check the number of parsed tokens. If it's not `27`, remove.  
- Removing 4 rows, and the whole data can be converted into `DataFrame`
- Inspect data
  - Check uniqueness
  - Check data types and do the type conversion. Since we read it using `csv.reader`, the data type is not auto-detected.
  - Upon checking data, I also remove a row with `annual_inc == 0`, because I assume that for the loan business, income should be required.
  - Save the result file

# 2) Data preparation & EDA
`02 Prep & EDA.ipynb`
## Explore each of the variables
 - Distribution, Outliers
 - In case of numerical variables, I also check the distributions for Full Paid/Charged Off groups.
 - In case of categorical variables, I also check the contingency table for Full Paid/Charged Off groups.
## Feature engineering
I generate many features, but in the end only two of them are used in the final model which are;
- `installment_inc_ratio`  
$$ \frac{MonthlyInstallment}{AnnualIncome/12} $$
Similar to DTI (debt-to-income), but according to the data dict, the requested loan amount is not included in the calculation. So I made this to measure the additional debt obligation to the borrower if the loan is accepted.
- `mort_acc_missing`  
Since 10% of the whole dataset of `mort_acc` (the number of mortgage accounts) are missing. So I imputed zero for missing values, and create a variable to tell that `mort_acc` is missing
## Categorical encoding
After going through all the variables, one-by-one. If a variable is considered as a categorical, then it is converted using pandas `category` data type. There are both `Nominal` and `Ordinal` categories in this dataset. For the ordinal category, I have to specify the order manually.  

**All categories** are encoded using `Ordinal` encoding with reasons;  
- I understand that `Nominal` categories should be encoded using One-Hot encoding (if its cardinality is not high), but it creates a lot of new features and tend to train the model slower with excessive features.  
- Tree-based models are capable of learning from Nominal categories even if it's encoded using Ordinal way.

# 3) Modeling

## Holdout test set
Load preprocessed dataset from earlier step and split it into `Train`, `Validation` and `Test` sets with 0.6/0.2/0.2 proportion respectively.

## Feature selection
- Check `Mutual Information Scores`  
To see predictive power of features using the non-parametric method. But it can only measure `discrete` variables
- Check ANOVA Test  
Just to see if there's a statistically difference in a `continuous` variable between Fully Paid/Charged Off group.  
It doesn't help much though. (some variable may not be normally distributed and that breaks the assumption of the test)
- Try fitting the model using `all features` using `XGBoost` and its default parameters. I Then check its `feature importance`
- All of the above steps are just for a guildline to help picking features manually

## Model Training
- Use `XGBoost` only (The original idea was to use RandomForst as a baseline, and try LightGBM, and other models)
- One way to address imbalanced problem using `XGBoost` is to specify `scale_pos_weight` parameter like this  
```Python
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
```
- `roc-auc` as the evaluation metrics.
- In previous step, I fitted the model using `all features`.
- Trying selecting `22 features` to train, and the roc-auc improves a bit. 
- Try resampling methods - `Under-sampling`, `Over-sampling` and `SMOTE`. None of them improves the performance.
- Hand-tune hyperparameter, the performance improves a bit.
- Final Model - use Train+Validation to train and it improves Test Set performance.  
I did make sure that I didn't use Test set in the `eval_set` parameter to make sure that test set is untouched

| Experiment | Train-AUC | Valid-AUC | Test-AUC | Note |
|---|---|---|---|---|
| All features, default parameters | 0.7586 | 0.7211 | 0.7156 | |
| Selected features, default params | 0.7544 | 0.7218 | 0.7169 | |
| Selected features, default params, Under-sampling | 0.7631 | 0.7201 | 0.7120 | |
| Selected features, default params, Over-sampling | 0.7527 | 0.7205 | 0.7142 | |
| Selected features, default params, SMOTE | 0.9475 | 0.6956 | 0.6921 | Clearly overfitted |
| Selected features, tuned params | 0.7422 | 0.7248 | 0.7190 | |
| Final Model | 0.7615 | N/A | 0.7202 | Best |

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





