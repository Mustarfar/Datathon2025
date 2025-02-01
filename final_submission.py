import re
import spacy
from fuzzywuzzy import fuzz
from collections import Counter
import networkx as nx
import torch
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import plotly.graph_objects as go
import random

filepath = "insert file path here"
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



def adjust_node_positions(pos, min_distance=0.2):
    """Adjusts node positions to ensure minimum spacing."""
    adjusted_pos = pos.copy()
    keys = list(adjusted_pos.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x1, y1 = adjusted_pos[keys[i]]
            x2, y2 = adjusted_pos[keys[j]]

            # Calculate distance
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # If too close, adjust positions
            if dist < min_distance:
                shift_x = (x2 - x1) * (min_distance / dist)
                shift_y = (y2 - y1) * (min_distance / dist)
                adjusted_pos[keys[j]] = (x2 + shift_x, y2 + shift_y)

    return adjusted_pos

def draw_final_no_duplicate_verbs_graph(relationships, entity_counts, max_relations=15):
    """Final optimized graph with spaced elements and no duplicate verbs."""
    G = nx.DiGraph()

    # Filter relationships to reduce clutter
    relation_freq = Counter(relationships)
    filtered_rels = [rel for rel, freq in relation_freq.most_common(max_relations)]

    # Add nodes and edges, but only label edges with objects
    for subj, verb, obj in filtered_rels:
        if obj:
            G.add_node(verb, type="verb")  # Verbs as nodes
            G.add_edge(subj, verb)  # Subject → Verb
            G.add_edge(verb, obj)  # Verb → Object

    # Use Fruchterman-Reingold layout with increased spacing
    pos = nx.fruchterman_reingold_layout(G, k=2.0, seed=42)  # k controls spacing

    # Apply minimum distance adjustment
    pos = adjust_node_positions(pos, min_distance=0.25)

    # Prepare edges (NO LABELS)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='gray'),
        hoverinfo='none',  # No labels
        mode='lines'
    )

    # Prepare nodes
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    max_freq = max(entity_counts.values(), default=1)

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)  # Show entity or verb
        node_size.append(22 if node in [rel[1] for rel in filtered_rels] else 12 + 18 * (entity_counts.get(node, 1) / max_freq))
        node_color.append('orange' if node in [rel[1] for rel in filtered_rels] else 'blue')  # Orange for verbs, blue for entities

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title=dict(text='Entity Frequency'),
                xanchor='left'
            )
        )
    )

    # Layout adjustments
    layout = go.Layout(
        title='Relationship between Entities within Dataset',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.show()



sample_text = df.values.tolist()[16][1]

cleaned_text = preprocess_text(sample_text)
print("Pre-processing Function:")
print(cleaned_text)
print()

# Process with NLP model
entities = extract_entities(cleaned_text)
print("Entity Extraction Function:")
print(entities)
print()

relations = filter_low_freq_relationships(extract_relationships(cleaned_text), min_freq=2)
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
# draw_network_graph(relations)
print()

# Compute entity frequency
entity_counts = dict(entity_frequencies(entities))


draw_final_no_duplicate_verbs_graph(relations, entity_counts, max_relations = 12)
