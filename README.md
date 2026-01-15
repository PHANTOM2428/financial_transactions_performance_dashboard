# Financial Transactions Performance Dashboard

A comprehensive analysis of customer transactions and credit profiles to identify target segments for the launch of AtliQo Bank’s credit card. Includes data cleaning, exploratory data analysis, visualizations, and insights on income, age, credit score, and transaction behavior.

---

## Table of Contents
1. Project Overview
2. Dataset
3. Data Cleaning
4. Exploratory Data Analysis
5. Key Insights
6. Target Market
7. Visualizations
8. Conclusion
9. Folder Structure

---

## Project Overview
This project analyzes AtliQo Bank’s customer demographics, credit profiles, and transaction behavior to identify the ideal target segment for a new credit card launch. The project focuses on data quality, business-driven cleaning, and actionable insights.

---

## Dataset
The analysis uses the following datasets:

- customers.csv – Customer demographics (age, gender, occupation, location, income)
- credit_profiles.csv – Credit score, credit limit, outstanding debt, credit utilization
- avg_transactions_after_campaign.csv – Aggregated transaction metrics
- transactions.zip – Transaction-level data (platform, product category, payment type, transaction amount)
- E_MasterCardDump.zip – Raw master dataset

Note: Large datasets are provided as zip files due to GitHub file size limits.

---

## Data Cleaning
- Handled missing values in annual_income using occupation-wise median
- Filled missing credit_limit using mode per credit score range
- Treated outliers in age, income, outstanding_debt, and transaction amount
- Standardized categorical variables and ensured dataset consistency

---

## Exploratory Data Analysis
The following analyses were performed:
- Age distribution and age-group segmentation
- Income distribution by occupation, gender, and location
- Credit score vs credit limit analysis
- Transaction behavior by platform, product category, and payment type
- Correlation analysis between income, credit score, and credit limit

---

## Key Insights
- Majority of customers belong to the 26–48 age group, while 18–25 represents ~26% of the base
- Credit limit has a strong positive correlation with credit score
- Younger customers show lower credit card usage
- Top spending categories for young customers include Electronics, Fashion & Apparel, and Beauty & Personal Care
- Amazon, Flipkart, and Alibaba are the most used platforms

---

## Target Market
The identified untapped target segment for the trial credit card launch:
- Age group: 18–25
- Average annual income below $50k
- Limited credit history and low credit card usage
- High spending on Electronics, Fashion & Apparel, and Beauty & Personal Care

---

## Visualizations
Key visualizations included in this project:
- Income distribution and income by occupation
- Age distribution before and after outlier treatment
- Credit score vs credit limit
- Payment type distribution across age groups
- Product category distribution by age group
- Correlation heatmap

All visualizations are available in the Images folder.



---

## Conclusion
This analysis identifies a promising untapped customer segment for AtliQo Bank’s credit card launch. The insights derived can guide marketing strategy, credit policy design, and customer acquisition planning.

---

## Folder Structure
