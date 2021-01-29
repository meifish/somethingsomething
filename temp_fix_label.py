import json

with open('./toy_dataset/compositional/labels.json', 'r') as labels:
    with open('./toy_dataset/compositional/labels_new.json', 'w') as new_labels:
        labels = json.load(labels)
        ascending_labels = {action : i+1 for (i, action) in enumerate(labels)}
        json.dump(ascending_labels, new_labels)