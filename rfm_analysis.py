import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(df):
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

def create_rfm(df):
    analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=2)
    rfm = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (analysis_date - x.max()).days, 'InvoiceNo': 'nunique', 'TotalPrice': 'sum'})
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm['R'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm['M'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str)
    seg_map = {r'[1-2][1-2]': 'hibernating', r'[1-2][3-4]': 'at_Risk', r'[1-2]5': 'cant_loose', r'3[1-2]': 'about_to_sleep', r'33': 'need_attention', r'[3-4][4-5]': 'loyal_customers', r'41': 'promising', r'51': 'new_customers', r'[4-5][2-3]': 'potential_loyalists', r'5[4-5]': 'champions'}
    rfm['Segment'] = rfm['RFM_Score'].replace(seg_map, regex=True)
    return rfm

def plot_rfm(rfm):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Segment', data=rfm, order=rfm['Segment'].value_counts().index, palette='viridis')
    plt.title('Customer Distribution by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)
df_cleaned = clean_data(df)
rfm_df = create_rfm(df_cleaned)

plot_rfm(rfm_df)