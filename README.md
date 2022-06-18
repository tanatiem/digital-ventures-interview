# Digital Ventures Interview
By: Tanat Iempreedee

As a part of job interview with Digital Ventures, please find my deliverables in this repository.
My main tool to develop the model is Python using Jupyter Notebook.

## Objective
Build a classfication model to score the probability of a borrower having `Fully Paid` loan status.
- Task: Binary Classification
- Labeling: Positive (1) - Fully Paid, Negative (0) - Charged Off
- Unit of analysis: Borrower

## Code Files
In order to keep the code not too long, I divided them into 3 parts
- `01 Clean.ipynb` is for data cleaning.
- `02 Prep & EDA.ipynb` is for basic EDA, feature engineering, and data preprocessing
- `03 Modeling.ipynb` is the main part where I split datasets, train the model, report model performance and interpretability.

Please note that each part produce output files after it's done. This is my personal habit to save state of the work.  
In case something crashes, I don't have to restart all over. But the file are too big to upload due to GitHub limitation, not all the files are uploaded.

# Data Cleaning
`01 Clean.ipynb`
- `pandas.read_csv` doesn't work with some corrupted records. Using `csv.reader` instead
- Identify the corrupted rows by simply check the number of parsed tokens. If it's not `27`, remove.  
- Removing 4 rows, and the whole data can be converted into `DataFrame`
- Inspect data
  - Check uniqueness
  - Check data types and do the type conversion. Since we read it using `csv.reader`, the data type is not auto-detected.
  - Upon checking data, I also remove a row with `annual_inc == 0`, because I assume that for the loan business, income should be required.
  - Save the result file

# Data preparation & EDA
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


