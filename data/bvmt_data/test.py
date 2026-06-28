import pandas as pd
df = pd.read_csv('bvmt_data/_all_news_combined.csv')
print(df[df['symbole'] == 'GIF'])