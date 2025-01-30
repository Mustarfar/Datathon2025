import re
import spacy
import fuzzywuzzy import fuzz
# Load NLP model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Clean and preprocess input text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]|\(.*?\)|[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    return text

sample_text = "Elon Musk is the CEO of Tesla and SpaceX. He founded Neuralink."
cleaned_text = preprocess_text(sample_text)
print(cleaned_text)

# Process with NLP model
doc = nlp(cleaned_text)


def extract_entities(text):
    """Extract named entities from text."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

entities = extract_entities(cleaned_text)
print(entities)

def extract_relationships(text):
    """Extract relationships between entities using dependency parsing."""
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
print(relations)

      from fuzzywuzzy import fuzz

def merge_entities(entities):
    """Merge entities with similar names."""
    merged = []
    for ent1 in entities:
        for ent2 in entities:
            if ent1 != ent2 and fuzz.ratio(ent1[0], ent2[0]) > 85:
                merged.append((ent1[0], ent2[1]))  # Keep first entity
    return list(set(merged))

merged_entities = merge_entities(entities)
print(merged_entities)

