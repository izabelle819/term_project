import pandas as pd
import numpy as np
import re
from bertopic import BERTopic
from hdbscan import HDBSCAN
from pickle import dump, load

print("imported libraries")

data = pd.read_csv("input_data.csv")
df = pd.DataFrame(data)

print("data in")

product_dict = {
    re.sub(r'[^A-Za-z0-9 ]', '', product)[:30]: group[['Rate', 'Summary', 'Sentiment']].reset_index(drop=True)
    for product, group in df.groupby('product_name')
}

filtered_products = {}
custom_hdbscan = HDBSCAN(min_cluster_size=2, min_samples=1)

for name, curr_product in product_dict.items():
    curr_product = curr_product.dropna(subset=['Summary'])
    curr_product = curr_product[curr_product['Summary'].apply(lambda x: len(str(x).split()) > 5)]

    reviews = curr_product['Summary'].tolist()

    if len(reviews) > 10:
        filtered_products[name] = curr_product.copy()

print("filtered data len: " + str(len(filtered_products.keys())))

product_topics = {}

for name, product in filtered_products.items():
    print(name + " being processed")
    summaries = product['Summary'].tolist()
    topic_model = BERTopic(language="english", hdbscan_model=custom_hdbscan)
    topics, _ = topic_model.fit_transform(summaries)
    
    product['Topic'] = topics
    product_topics[name] = {
        "data": product,
        "model": topic_model
    }

    print(len(summaries))

print(len(product_topics))
'''
with open("product_topics.pkl", "wb") as f:
    dump(product_topics, f)

first_product = next(iter(product_topics.values()))
print(first_product["model"].get_topic_info())

topic_model.visualize_topics()



df['clean_product'] = df['product_name'].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]', '', x)[:30])

# Step 2: Filter reviews with more than 5 words
df['review_word_count'] = df['Review'].apply(lambda x: len(str(x).split()))
df = df[df['review_word_count'] > 5]

# Step 3: Create dictionary of DataFrames
product_dict = {
    name: group[['Rate', 'Summary', 'Sentiment', 'Review']].reset_index(drop=True)
    for name, group in df.groupby('clean_product')
}

print(len(product_dict.keys()))
print(product_dict.keys())

product_topics = {}

for product_name, product_df in product_dict.items():
    reviews = product_df['Review'].dropna().tolist()

    if len(reviews) < 5:
        print(f"Skipping {product_name}: not enough reviews.")
        continue

    print(f"Training BERTopic for: {product_name}")

    # Initialize BERTopic — can specify embedding_model=embedding_model if using sentence transformers
    topic_model = BERTopic(language="english", verbose=False)
    topics, _ = topic_model.fit_transform(reviews)

    # Add topics back to the product's DataFrame
    product_df['Topic'] = topics

    # Store the result
    product_topics[product_name] = {
        'data': product_df,
        'model': topic_model
    }

# 🧪 Example: Show top 5 topics for one product
example = list(product_topics.keys())[0]
print(f"\nTop topics for {example}:\n")
print(product_topics[example]['model'].get_topic_info().head())

'''