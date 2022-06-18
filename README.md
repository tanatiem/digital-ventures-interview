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

## 
