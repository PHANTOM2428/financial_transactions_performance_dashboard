#!/usr/bin/env python
# coding: utf-8

# In[9]:


import mysql.connector
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Ayush@5387",
    database="e_master_card"
)


# In[10]:


df_cust = pd.read_sql("SELECT * FROM customers", conn)
print(df_cust.head())


# In[ ]:


df_transactions = pd.read_sql("SELECT * FROM transactions", conn)
print(df_transactions.head())


# In[12]:


df_cs = pd.read_sql("SELECT * FROM credit_profiles", conn)
print(df_cs.head())


# In[13]:


df_cust.describe()


# In[14]:


df_cust.groupby("occupation")["annual_income"].median()


# In[15]:


df_cust.iloc[[14,82]]


# In[16]:


df_cust["annual_income"] = df_cust["annual_income"].replace(0, np.nan)


# In[17]:


df_cust.isnull().sum()


# In[18]:


occupation_wise_income_median = df_cust.groupby("occupation")["annual_income"].median()
occupation_wise_income_median


# In[19]:


def get_median_val(row):
    if pd.isnull(row["annual_income"]):
        return occupation_wise_income_median.get(row["occupation"], np.nan)
    else:
        return row["annual_income"]

df_cust["annual_income"] = df_cust.apply(get_median_val, axis=1)

print(df_cust)


# In[20]:


df_cust.iloc[[14,82]]


# In[21]:


df_cust.isnull().sum()


# In[22]:


sns.histplot(df_cust["annual_income"], kde = True)
plt.show()


# In[23]:


df_cust.describe()


# In[24]:


df_cust[df_cust["annual_income"]<100]


# In[25]:


for index, row in df_cust.iterrows():
    if row["annual_income"]<100:
        df_cust.at[index, "annual_income"]= occupation_wise_income_median[row["occupation"]]


# In[26]:


df_cust[df_cust["annual_income"]<100]


# In[27]:


avg_income_per_occupation = df_cust.groupby("occupation")["annual_income"].mean()
avg_income_per_occupation


# In[28]:


avg_income_per_occupation


# In[29]:


avg_income_per_occupation.values


# In[30]:


sns.barplot(x=avg_income_per_occupation.index,y = avg_income_per_occupation.values, palette = "tab10")
plt.xticks(rotation = 45)
plt.show()


# In[31]:


categorial_columns = ["gender", "location", "occupation" , "marital_status"]
for col in categorial_columns:
    avg_income_per_group = df_cust.groupby(col)["annual_income"].mean().sort_values()
    sns.barplot(x = avg_income_per_group.index, y = avg_income_per_group.values, palette = "tab10")
    plt.xticks(rotation = 45)
    plt.show()


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns

categorial_columns = ["gender", "location", "occupation", "marital_status"]

# Create 2 rows and 2 columns of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Adjust size as needed

# Flatten axes for easier iteration
axes = axes.flatten()

for i, col in enumerate(categorial_columns):
    avg_income_per_group = df_cust.groupby(col)["annual_income"].mean().sort_values()
    sns.barplot(x=avg_income_per_group.index,
                y=avg_income_per_group.values,
                palette="tab10",
                ax=axes[i])
    axes[i].set_title(f"Average Income by {col.capitalize()}")
    axes[i].set_ylabel("Average Income")
    axes[i].set_xlabel(col.capitalize())
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# # Analyze age column

# In[33]:


df_cust.isnull().sum()


# In[34]:


df_cust.age.describe()


# In[35]:


sns.histplot(df_cust.age, bins = 20)


# In[36]:


outliers = df_cust[(df_cust.age<10) | (df_cust.age >80)]
outliers


# In[37]:


median_age_per_occupation = df_cust.groupby("occupation")["age"].median()
median_age_per_occupation


# In[38]:


for index, row in outliers.iterrows():
    df_cust.at[index, "age"]= median_age_per_occupation[row["occupation"]]


# In[39]:


df_cust[(df_cust.age<10) | (df_cust.age >80)]


# In[40]:


sns.histplot(df_cust.age, bins = 20)


# In[41]:


df_cust.head()


# In[42]:


# Define bins and labels
bins = [17, 25, 48, 70]
labels = ['18-25', '26-48', '49-70']

# Add the new column
df_cust['age_group'] = pd.cut(df_cust['age'], bins=bins, labels=labels)
df_cust.head()


# In[43]:


age_group_counts = df_cust.age_group.value_counts(normalize = True)*100
age_group_counts


# In[44]:


plt.pie(age_group_counts,labels = age_group_counts.index, shadow = True, autopct = "%1.1f%%", explode = (0.1, 0, 0), startangle = 140)
plt.title("Age distribution")
plt.legend()
plt.show()


# In[45]:


customer_location_gender = df_cust.groupby(["location" , "gender"]).size().unstack()
customer_location_gender


# In[46]:


customer_location_gender.plot(kind = "bar", stacked = "true")
plt.xticks(rotation = 45)
plt.show()


# # Explore credit score data
# 

# In[47]:


df_cs.head()


# In[48]:


df_cs.shape


# In[49]:


df_cs["cust_id"].nunique()


# In[50]:


df_cs[df_cs["cust_id"].duplicated(keep = False)]


# In[51]:


df_cs_clean_1 = df_cs.drop_duplicates(subset = "cust_id", keep = "last")
df_cs_clean_1


# In[52]:


df_cs_clean_1.isnull().sum()


# In[53]:


df_cs_clean_1[df_cs_clean_1.credit_limit.isnull()]  


# In[54]:


df_cs_clean_1.credit_limit.value_counts()


# In[55]:


plt.figure(figsize = (20,5))
plt.scatter(df_cs_clean_1.credit_limit, df_cs_clean_1.credit_score)
plt.xlabel("credit limit")
plt.ylabel("credit score")
plt.grid(True)
plt.show()


# In[56]:


bin_ranges = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
bin_labels = [f'{start}-{end-1}' for start, end in zip(bin_ranges, bin_ranges[1:])]
df_cs_clean_1["credit_score_range"] = pd.cut(df_cs_clean_1['credit_score'], bins=bin_ranges, labels=bin_labels, include_lowest=True, right=False)
df_cs_clean_1.head()


# In[57]:


mode_df = df_cs_clean_1.groupby('credit_score_range')['credit_limit'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None
).reset_index()
mode_df


# In[58]:


# Merge the mode values back with the original DataFrame
df_cs_clean_2 = pd.merge(df_cs_clean_1, mode_df, on='credit_score_range', suffixes=('', '_mode'))
df_cs_clean_2.sample(3)


# In[59]:


df_cs_clean_2 = pd.merge(df_cs_clean_1, mode_df, on='credit_score_range', suffixes=('', '_mode'))
df_cs_clean_2.sample(3)


# In[60]:


df_cs_clean_2[df_cs_clean_2.credit_limit.isnull()].sample(3)


# In[61]:


df_cs_clean_3 = df_cs_clean_2.copy()
df_cs_clean_3["credit_limit"].fillna(df_cs_clean_3["credit_limit_mode"], inplace = True)
df_cs_clean_3.isnull().sum()


# In[62]:


df_cs_clean_3.describe()


# In[63]:


df_cs_clean_3[df_cs_clean_3.outstanding_debt>df_cs_clean_3.credit_limit]


# In[64]:


df_cs_clean_3.loc[df_cs_clean_3['outstanding_debt'] > df_cs_clean_3['credit_limit'], 'outstanding_debt'] = df_cs_clean_3.credit_limit


# In[65]:


df_cs_clean_3[df_cs_clean_3.outstanding_debt>df_cs_clean_3.credit_limit]


# In[66]:


df_cust.head()


# In[67]:


df_cs_clean_3.head()


# In[68]:


df_merged = df_cust.merge(df_cs_clean_3, on ="cust_id", how = "inner")
df_merged.head()


# In[69]:


numerical_cols = ['credit_score', 'credit_utilisation', 'outstanding_debt', 'credit_limit', 'annual_income', 'age']
correlation_matrix = df_merged[numerical_cols].corr()
correlation_matrix


# In[70]:


plt.figure(figsize=(5, 3))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.8)
plt.title('Correlation Plot')
plt.show()


# In[71]:


df_transactions.head()


# In[72]:


df_transactions.isnull().sum()


# In[73]:


df_transactions[df_transactions.platform.isnull()]


# In[74]:


df_transactions.platform.mode()


# In[75]:


sns.countplot(y = "product_category",hue = "platform", data = df_transactions)
plt.show()


# In[76]:


df_transactions.platform.fillna(df_transactions.platform.mode()[0], inplace = True)


# In[77]:


df_transactions.isnull().sum()


# In[78]:


df_transactions.describe()


# In[79]:


df_trans_zero = df_transactions[df_transactions.tran_amount==0]
df_trans_zero


# In[80]:


df_trans_zero[["platform", "product_category", "payment_type"]].value_counts()


# In[81]:


df_trans_1 = df_transactions[(df_transactions.platform=='Amazon')&(df_transactions.product_category=='Electronics')&(df_transactions.payment_type=='Credit Card')]
df_trans_1.head()


# In[82]:


df_trans_1[df_trans_1.tran_amount>0].head()


# In[83]:


median_to_replace = df_trans_1[df_trans_1.tran_amount>0].tran_amount.median()
median_to_replace


# In[84]:


df_transactions['tran_amount'].replace(0, median_to_replace, inplace = True)
df_transactions.describe()


# In[85]:


sns.histplot(df_transactions[df_transactions.tran_amount<10000].tran_amount, bins = 30)
plt.show()


# In[86]:


q1, q3 = df_transactions.tran_amount.quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 2*iqr
upper


# In[87]:


df_trans_outliers = df_transactions[df_transactions.tran_amount>upper]
df_trans_outliers.head()


# In[88]:


tran_mean_per_category = df_transactions[df_transactions.tran_amount<upper].groupby("product_category")["tran_amount"].mean()
tran_mean_per_category


# In[89]:


df_trans_normal = df_transactions[df_transactions.tran_amount<upper]
df_trans_normal


# In[90]:


df_transactions.loc[df_trans_outliers.index, 'tran_amount'] = df_trans_outliers['product_category'].map(tran_mean_per_category)


# In[91]:


df_transactions.loc[df_trans_outliers.index]


# In[92]:


sns.histplot(df_transactions.tran_amount, kde = True, bins = 20)
plt.show()


# In[93]:


df_transactions.head()


# In[94]:


sns.countplot(x=df_transactions.payment_type, stat='percent')
plt.show()


# In[95]:


df_merged_2 = df_merged.merge(df_transactions, on = "cust_id", how = "inner")
df_merged_2.head()


# In[98]:


fig, ax2 = plt.subplots(figsize=(10, 6))

sns.countplot(
    x='age_group',
    hue='platform',
    data=df_merged_2,
    palette='Set3',
    ax=ax2
)

ax2.set_title("Platform Count By Age Group")
ax2.set_xlabel("Age Group")
ax2.set_ylabel("Count")
ax2.legend(title="Platform", loc='upper right')

plt.show()


# In[97]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

sns.countplot(x='age_group', hue="product_category", data=df_merged_2, ax=ax1)
ax1.set_title("Product Category Count By Age Group")
ax1.set_xlabel("Age Group")
ax1.set_ylabel("Count")
ax1.legend(title="Product Category", loc='upper right')

sns.countplot(x='age_group', hue="platform", data=df_merged_2, ax=ax2)
ax2.set_title("Platform Count By Age Group")
ax2.set_xlabel("Age Group")
ax2.set_ylabel("Count")
ax2.legend(title="Product Category", loc='upper right')

plt.show()


# In[99]:


# List of categorical columns
cat_cols = ['payment_type', 'platform', 'product_category', 'marital_status', 'age_group']

num_rows = 3
# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))

# Flatten the axes array to make it easier to iterate
axes = axes.flatten()

# Create subplots for each categorical column
for i, cat_col in enumerate(cat_cols):
    # Calculate the average annual income for each category
    avg_tran_amount_by_category = df_merged_2.groupby(cat_col)['tran_amount'].mean().reset_index()

    # Sort the data by 'annual_income' before plotting
    sorted_data = avg_tran_amount_by_category.sort_values(by='tran_amount', ascending=False)

    sns.barplot(x=cat_col, y='tran_amount', data=sorted_data, ci=None, ax=axes[i], palette='tab10')
    axes[i].set_title(f'Average transaction amount by {cat_col}')
    axes[i].set_xlabel(cat_col)
    axes[i].set_ylabel('Average transaction amount')

    # Rotate x-axis labels for better readability
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

# Hide any unused subplots
for i in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.show()


# In[101]:


# Create age-group level aggregated metrics
age_group_metrics = (
    df_merged_2
    .groupby('age_group')
    .agg(
        annual_income=('annual_income', 'mean'),
        credit_limit=('credit_limit', 'mean'),
        credit_score=('credit_score', 'mean')
    )
    .reset_index()
)

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1: Average Annual Income by Age Group
sns.barplot(
    x='age_group',
    y='annual_income',
    data=age_group_metrics,
    palette='tab10',
    ax=ax1
)
ax1.set_title('Average Annual Income by Age Group')
ax1.set_xlabel('Age Group')
ax1.set_ylabel('Average Annual Income')

# Plot 2: Average Credit Limit by Age Group
sns.barplot(
    x='age_group',
    y='credit_limit',
    data=age_group_metrics,
    palette='hls',
    ax=ax2
)
ax2.set_title('Average Credit Limit by Age Group')
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Average Credit Limit')

# Plot 3: Average Credit Score by Age Group
sns.barplot(
    x='age_group',
    y='credit_score',
    data=age_group_metrics,
    palette='viridis',
    ax=ax3
)
ax3.set_title('Average Credit Score by Age Group')
ax3.set_xlabel('Age Group')
ax3.set_ylabel('Average Credit Score')

plt.tight_layout()
plt.show()

