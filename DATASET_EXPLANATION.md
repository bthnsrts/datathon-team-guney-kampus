# Dataset Description

## Files Overview

- **customer_history.csv**: Contains monthly aggregated transaction history of customers (EFT, credit card expenses, etc.)
- **customers.csv**: Contains demographic information of customers (age, gender, etc.)
- **reference_data.csv**: Shows which customers are labeled as "churn" on which reference date, used for training the model
- **reference_data_test.csv**: Contains reference information of test customers for whom churn prediction is requested
- **sample_submission.csv**: Shows the format in which predictions should be submitted to the competition

## Column Descriptions

### customer_history.csv

| Column Name | Description |
| ----------- | ----------- |
| cust_id | Unique customer number |
| date | Date of the month in which transactions are aggregated (in YYYY-MM-DD format). Each row represents a summary of transactions for that month. |
| mobile_eft_all_cnt | Number of mobile electronic fund transfer (EFT) transactions by the customer in the relevant month |
| active_product_category_nbr | Number of active product categories of the customer in the relevant month |
| mobile_eft_all_amt | Total amount of mobile EFT transactions by the customer in the relevant month (€) |
| cc_transaction_all_amt | Total amount of credit card transactions by the customer in the relevant month (€) |
| cc_transaction_all_cnt | Number of credit card transactions by the customer in the relevant month |

### customers.csv

| Column Name | Description |
| ----------- | ----------- |
| cust_id | Unique customer number |
| gender | Customer gender |
| age | Customer age in years |
| province | Region code where the customer resides |
| religion | Customer's religion |
| work_type | Customer's employment status |
| work_sector | Customer's industry sector |
| tenure | Customer relationship duration (in months) |

### reference_data.csv

| Column Name | Description |
| ----------- | ----------- |
| cust_id | Unique customer number |
| ref_date | Reference date on which the customer churn label was assigned |
| churn | Customer churn status within 6 months after the reference date (1 = churn occurred, 0 = churn did not occur) |

### reference_data_test.csv

| Column Name | Description |
| ----------- | ----------- |
| cust_id | Unique customer number |
| ref_date | Reference date on which the customer churn label was assigned |

### sample_submission.csv

| Column Name | Description |
| ----------- | ----------- |
| cust_id | Unique customer number |
| churn | Customer churn status within 6 months after the reference date (1 = churn occurred, 0 = churn did not occur) |
