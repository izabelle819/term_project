import pandas as pd
import re

data = pd.read_csv("input_data.csv")
df = pd.DataFrame(data)

print(f"number of review... {len(df)}")

product_dict = {
    re.sub(r'[^A-Za-z0-9 ]', '', product)[:30]: group[['product_name','Rate', 'Summary', 'Sentiment']].reset_index(drop=True)
    for product, group in df.groupby('product_name')
}

print(f"number of products... {len(product_dict.keys())}")

