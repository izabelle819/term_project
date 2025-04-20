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
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_topic_scores(topic_scores, topic_names, product_name):
    labels = [topic_names[k] for k in topic_scores.keys()]
    scores = [topic_scores[k] for k in topic_scores.keys()]
    plt.figure(figsize=(12, 7))
    sns.barplot(x=scores, y=labels, palette="coolwarm", orient='h')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title(f"Sentiment-Weighted Topic Scores\nProduct: {product_name}")
    plt.xlabel("Normalized Sentiment Score")
    plt.ylabel("Topics")
    plt.tight_layout()
    safe_name = re.sub(r'[^A-Za-z0-9]', '_', product_name)[:30]
    plt.savefig(f"plots/{safe_name}_topic_scores.png")
    plt.close()


def pretty_print(dict_in, names):
    for key in sorted(dict_in.keys()):
        print(f"{key}, {names[key]}: {dict_in[key]}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

client = openai.OpenAI(api_key=api_key.key_)
ai_model = OpenAI(client, model="gpt-4o-mini", chat=True)

print("all libraries imported")

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
# MAKE SURE TO SHARE SHEET TO: gspread-access@sheets-access-456721.iam.gserviceaccount.com
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
spreadsheets = client.openall()
for sheet in spreadsheets:
    print(sheet.title)
sheet = client.open("Responses").sheet1 # CHECK IF THE SHEET NAME IS CORRECT
data = sheet.get_all_records() 
df = pd.DataFrame(data)

print(df)

print("data in")

filtered_products = {"ECE49595: NLP": df}

print("filtered data len: " + str(len(filtered_products.keys())))

f = open("live_demo.pkl", "wb")
dump(len(filtered_products.keys()), f)
product_topics = {}
i = 0

key_model = KeyBERTInspired()

mmr_model = MaximalMarginalRelevance(diversity=0.3)

client = openai.OpenAI(api_key=api_key.key_)
ai_model = OpenAI(client, model="gpt-4o-mini", chat=True)

rep_models = [key_model, mmr_model]

custom_hdbscan = HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True)

for name, product in filtered_products.items():
    print("\n  --------------------- \n")
    print(name + " being processed")
    summaries = product['Review'].tolist()
    sentiments = product['Rating'].tolist()
    topic_model = BERTopic(language="english", hdbscan_model=custom_hdbscan, representation_model=rep_models)

    topics, _ = topic_model.fit_transform(summaries)
    topic_model.reduce_topics(summaries, nr_topics="auto")
    topics, _ = topic_model.transform(summaries)

    num_of_pos = sentiments.count(4) + sentiments.count(5)
    num_of_neg =  sentiments.count(1) + sentiments.count(2)
    ratings_normalized = [int(x) - 3 for x in product['Rating'].tolist()]

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

    pretty_print(topic_scores, topic_names)
    
    os.makedirs("plots", exist_ok=True)
    visualize_topic_scores(topic_scores, topic_names, name)

    for topic_id in sorted(product['Topic'].unique()):
        topic_reviews = product[product['Topic'] == topic_id]['Review'].dropna().tolist()
        if len(topic_reviews) == 0:
            continue
        joined_reviews = "\n".join(topic_reviews[:10])
        prompt = f"Summarize the key points, feedback, or issues raised in the following student course reviews:\n{joined_reviews}\n\n Summary:"
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in summarizing student feedback on university courses. Give all replies in short paragraph form."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message.content.strip()
            print(f"\n - Topic {topic_id} ({topic_names.get(topic_id, 'Unknown')}):\n{summary}\n")
        except Exception as e:
            print(f"Failed to generate summary for topic {topic_id}: {e}")
    data = {"product data": product, "topic info": topic_info, "topic names": topic_names, "topic scores": topic_scores}
    dump(data, f)

    # Clear memory
    del topic_model
    gc.collect()

f.close()

