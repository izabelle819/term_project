from bertopic import BERTopic

print("importing - past bert")

import pandas as pd
import numpy as np
import re
from hdbscan import HDBSCAN
from pickle import dump, load
import gc
import os
from bertopic.representation import KeyBERTInspired, OpenAI, MaximalMarginalRelevance
import api_key
import openai

def pretty_print(dict_in, names):
    for key in sorted(dict_in.keys()):
        print(f"{key}, { names[key] }: {dict_in[key]}")
        #print(f"{key}: {dict_in[key]}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

print("all libraries imported")

data = pd.read_csv("input_data.csv")
df = pd.DataFrame(data)

print("data in")

product_dict = {
    re.sub(r'[^A-Za-z0-9 ]', '', product)[:30]: group[['product_name','Rate', 'Summary', 'Sentiment']].reset_index(drop=True)
    for product, group in df.groupby('product_name')
}

print(f"{len(product_dict.keys())}")

filtered_products = {}

for name, curr_product in product_dict.items():
    curr_product = curr_product.dropna(subset=['Summary'])
    curr_product = curr_product[curr_product['Summary'].apply(lambda x: len(str(x).split()) > 10)]

    reviews = curr_product['Summary'].tolist()
    sentiments = curr_product['Sentiment'].tolist()

    if (len(reviews) >= 20) and (len([x for x in sentiments if x == "negative"]) > 10):
        filtered_products[name] = curr_product.copy()

print("filtered data len: " + str(len(filtered_products.keys())))

f = open("product_topics.pkl", "wb")
product_topics = {}
i = 0

key_model = KeyBERTInspired()

mmr_model = MaximalMarginalRelevance(diversity=0.3)

client = openai.OpenAI(api_key=api_key.key_)
ai_model = OpenAI(client, model="gpt-4o-mini", chat=True)

rep_models = [key_model, mmr_model]

custom_hdbscan = HDBSCAN(min_cluster_size=5, min_samples=2, prediction_data=True)

for name, product in filtered_products.items():
    print("\n  --------------------- \n")
    print(name + " being processed")
    summaries = product['Summary'].tolist()
    sentiments = product['Sentiment'].tolist()
    topic_model = BERTopic(language="english", hdbscan_model=custom_hdbscan, representation_model=rep_models)

    topics, _ = topic_model.fit_transform(summaries)
    topic_model.reduce_topics(summaries, nr_topics="auto")
    topics, _ = topic_model.transform(summaries)

    num_of_pos = sentiments.count("positive")
    num_of_neg = sentiments.count("negative")
    ratings_normalized = [int(x) - 3 for x in product['Rate'].tolist()]

    product['Topic'] = topics
    product_topics[name] = {
        "data": product,
        "model": topic_model
    }

    topic_scores = {}
    for topic in topics:
        topic_scores[topic] = 0

    for summary, topic, sentiment, rating in zip(summaries, topics, sentiments, ratings_normalized):
        #print(f"Summary: {summary[:60]}... → Topic: {topic} → Sentiment: {sentiment}")
        num = num_of_neg if sentiment == "negative" else num_of_pos
        topic_scores[topic] += rating / num

    topic_info = topic_model.get_topic_info()
    topic_names = topic_info.set_index("Topic")["Name"].to_dict()

    #print(f"{name} done, {len(summaries)} summaries")
    #print(topic_model.get_topic_info())
    pretty_print(topic_scores, topic_names)

    # Save intermediate results
    #dump(product_topics, f)

    # Clear memory
    del topic_model
    gc.collect()

