import json

from datasets import load_dataset

ds = load_dataset("strova-ai/hr-policies-qa-dataset", split="train")
#print(ds[0]["messages"][1], "\n\n", ds[0]["messages"][2])

docs = []
for i in range(len(ds)):
    doc = f"Question : {ds[i]["messages"][1]['content']} \n" \
          f"Answer : {ds[i]["messages"][2]['content']}" \

    docs.append(doc)

#print(docs[0])
            

ds_it = load_dataset("lvogelFH/synthetic-it-callcenter-tickets", split="train")
for i in range(len(ds_it)):
    if (ds_it[i]["content"] is None) or (ds_it[i]["close_notes"] is None):
        continue
    doc = f"Question : {ds_it[i]["content"]} \n" \
          f"Answer : {ds_it[i]["close_notes"]}" \

    docs.append(doc)

#json_string = json.dumps(docs)
#with open("sample.json", "w") as f:
#    f.write(json_string)

with open("sample.json", "r") as f:
    data = json.load(f)

print(data[0])

'''
print(ds_it[0]["short_description"])
print("======================")
print(ds_it[0]["content"])
print("======================")
print(ds_it[0]["close_notes"])
print("======================")
print(ds_it[0]["agent"])
print("======================")
'''

