import re
import spacy
from fuzzywuzzy import fuzz
from collections import Counter
import networkx as nx
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd

filepath = "/home/brandon/Datasets/Datathon_2025/news_excerpts_parsed.xlsx"
df = pd.read_excel(filepath)


# Load NLP model
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_lg") # or "en_core_web_trf" for transformer-based parsing




def preprocess_text(text):
    # Remove unnecessary spaces, but keep original case for better entity detection
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9\s\[\]\(\)]', '', text)  
    return text

def extract_entities(text):
    # """Extract named entities from text."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_relationships(text):
    """Enhanced extraction of relationships between entities using dependency parsing."""
    doc = nlp(text)
    relationships = []

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            subject = " ".join([t.text for t in token.subtree if t.dep_ in ("compound", "nsubj")])
            if subject.lower() in STOP_WORDS:  # Skip stopwords
                continue

            verb = token.head.text
            obj = [child.text for child in token.head.children if child.dep_ == "dobj"]
            obj = obj[0] if obj else ""

            if obj.lower() in STOP_WORDS:  # Skip stopwords
                continue

            relationships.append((subject, verb, obj))

        # Handle prepositional phrases
        if token.dep_ == "prep" and token.head.pos_ == "VERB":
            prep = token.text
            obj = [child.text for child in token.children if child.dep_ == "pobj"]
            obj = obj[0] if obj else ""

            if obj.lower() in STOP_WORDS:  # Skip stopwords
                continue

            relationships.append((token.head.text, prep, obj))

    return relationships

def filter_low_freq_relationships(relationships, min_freq=2):
    """Remove relationships that appear fewer than min_freq times."""
    freq = Counter(relationships)
    
    if all(count < min_freq for count in freq.values()):  # Prevent blank graphs
        print("⚠️ Warning: No relationships meet the minimum frequency threshold. Lowering min_freq to 1.") # can remove for aesthetic reasons
        min_freq = 1
    
    return [rel for rel in relationships if freq[rel] >= min_freq]

def merge_entities(entities):
    # """Merge entities with similar names."""
    merged = []
    for ent1 in entities:
        for ent2 in entities:
            if ent1 != ent2 and fuzz.ratio(ent1[0], ent2[0]) > 85:
                merged.append((ent1[0], ent2[1]))  # Keep first entity
    return list(set(merged))

def entity_frequencies(entities):
    """Count entity occurrences."""
    counts = Counter([ent[0] for ent in entities])
    return counts.most_common()

def draw_network_graph(relationships):
    """Create a cleaner and more readable network graph."""
    G = nx.DiGraph()

    for subj, verb, obj in relationships:
        if obj:
            G.add_edge(subj, obj, label=verb)

    plt.figure(figsize=(12, 8))  # Bigger figure for better readability
    pos = nx.spring_layout(G, k=0.8)  # Adjust spacing for clarity

    # Set node sizes based on frequency
    node_size = {node: 3000 if node in ["Meta", "Facebook", "EU", "law"] else 1500 for node in G.nodes()}
    
    # Color important nodes
    node_colors = ["lightcoral" if node in ["Meta", "Facebook", "EU"] else "lightblue" for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_size=[node_size[node] for node in G.nodes()],
            node_color=node_colors, edge_color="gray", font_size=10, font_weight="bold")

    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=9)

    plt.show()


sample_text = df.values.tolist()[2][1]

cleaned_text = preprocess_text(sample_text)
print("Pre-processing Function:")
print(cleaned_text)
print()

# Process with NLP model
entities = extract_entities(cleaned_text)
print("Entity Extraction Function:")
print(entities)
print()

relations = filter_low_freq_relationships(extract_relationships(sample_text), min_freq=2)
print("Relationship Extractor Function:")
print(relations)
print()

print("Entity Merging Function:")
merged_entities = merge_entities(entities)
print(merged_entities)
print()

print("Entity Frequency Function:")
print(entity_frequencies(entities))
print()

print("Drawing of Graphs:")
draw_network_graph(relations)
print()
