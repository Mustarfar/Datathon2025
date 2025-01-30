import re
import spacy
from fuzzywuzzy import fuzz
from collections import Counter
import networkx as nx
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()



# Load NLP model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # """Clean and preprocess input text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]|\(.*?\)|[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    return text

sample_text = "Elon Musk is the CEO of Tesla and SpaceX. He founded Neuralink."
cleaned_text = preprocess_text(sample_text)
print("Function 1:")
print(cleaned_text)
print()

# Process with NLP model
doc = nlp(cleaned_text)


def extract_entities(text):
    # """Extract named entities from text."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

entities = extract_entities(cleaned_text)
print("Function 2:")
print(entities)
print()

def extract_relationships(text):
    # """Extract relationships between entities using dependency parsing."""
    doc = nlp(text)
    relationships = []

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            obj = [child.text for child in token.head.children if child.dep_ == "dobj"]
            obj = obj[0] if obj else ""
            relationships.append((subject, verb, obj))
    
    return relationships

relations = extract_relationships(cleaned_text)
print("Function 3:")
print(relations)
print()

def merge_entities(entities):
    # """Merge entities with similar names."""
    merged = []
    for ent1 in entities:
        for ent2 in entities:
            if ent1 != ent2 and fuzz.ratio(ent1[0], ent2[0]) > 85:
                merged.append((ent1[0], ent2[1]))  # Keep first entity
    return list(set(merged))

print("Function 4:")
merged_entities = merge_entities(entities)
print(merged_entities)
print()

def entity_frequencies(entities):
    """Count entity occurrences."""
    counts = Counter([ent[0] for ent in entities])
    return counts.most_common()

print("Function 5:")
print(entity_frequencies(entities))
print()

def draw_network_graph(relationships):
    """Create a network graph from extracted relationships."""
    G = nx.Graph()
    
    for subj, verb, obj in relationships:
        if obj:  # Avoid empty objects
            G.add_edge(subj, obj, label=verb)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


print("Function 6:")
# draw_network_graph(relations)
print()
