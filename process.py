from pickle import load, dump
from openai import OpenAI
import api_key as key

#data = {"product data": product, " topic info": topic_info, "topic names": topic_names, "topic scores": topic_scores}

f = open("product_topics.pkl", "rb")
client = OpenAI(api_key = key.key_)

sys = """You are an expert product review summarizer. The data you get in is the name of the product, 
        a list of identified topics, and associated sentiment scores, calculated algorithmically based on a normalized rating system 
        given the density of positive/negative reviews. You are responsible for outputting nothing but a summarizaiton of this data 
        in text form. This should be a short paragraph clearly outlining positives and negatives. Group related topics together 
        where possible. Write as if the reader will have no insight into the given data, i.e. dont mention the sentiment scores"""

data_list = []
products = []
topic_info = []
topic_names = []
topic_scores = []

for i in range(load(f)):
    in_data = load(f)
    data_list.append(in_data)
    products.append(in_data["product data"])
    topic_info.append(in_data["topic info"])
    topic_names.append(in_data["topic names"])
    topic_scores.append(in_data["topic scores"])

print(len(data_list))
print(len(topic_info))
print(len(topic_names))
print(len(topic_scores))

descriptions = []

for p, names, score in zip(products, topic_names, topic_scores):
    product_name = p["product_name"].iloc[0]
    gpt_info = f"Product Name: {product_name[:50]} \n"
    for key in sorted(names.keys()):
        gpt_info += f"\t {names[key]}: {score[key]}\n"

    print(gpt_info)

    message_log = [ 
        {
            "role" : "system",
            "content" : sys
        },
        {
            "role" : "user",
            "content" : gpt_info
        }
    ]

    completion = client.chat.completions.create(model="gpt-4o", messages=message_log)
    summary = completion.choices[0].message.content
    descriptions.append({"product": product_name, "summary": summary})
    print(f"\nSummary for {product_name[:30]}...\n{summary}\n")

'''with open("gpt_descriptions.pkl", "wb") as f:
    dump(descriptions, f)'''