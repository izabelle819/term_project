from bertopic import BERTopic
import pandas as pd
from hdbscan import HDBSCAN
from pickle import dump, load
import gc
import os
from bertopic.representation import KeyBERTInspired, OpenAI, MaximalMarginalRelevance
import api_key
import openai

client = openai.OpenAI(api_key=api_key.key_)
ai_model = OpenAI(client, model="gpt-4o-mini", chat=True)

def pretty_print(dict_in, names):
    for key in sorted(dict_in.keys()):
        print(f"{key}, {names[key]}: {dict_in[key]}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

data = pd.read_csv("responses.csv")
df = pd.DataFrame(data)

filtered_products = {"ECE49494: NLP": data}

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
        num = num_of_neg if sentiment == "negative" else num_of_pos
        topic_scores[topic] += rating / num

    topic_info = topic_model.get_topic_info()
    topic_names = topic_info.set_index("Topic")["Name"].to_dict()

    pretty_print(topic_scores, topic_names)

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
                    {"role": "system", "content": "You are an expert in summarizing student feedback on university courses."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message.content.strip()
            print(f"\n - Topic {topic_id} ({topic_names.get(topic_id, 'Unknown')}):\n{summary}\n")
        except Exception as e:
            print(f"Failed to generate summary for topic {topic_id}: {e}")
    data = {"product data": product, "topic info": topic_info, "topic names": topic_names, "topic scores": topic_scores}
    dump(data, f)
    del topic_model
    gc.collect()
f.close()
