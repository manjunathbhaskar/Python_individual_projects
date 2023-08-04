#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install nltk')


# In[23]:


import os
os.getcwd()


# In[20]:


import PyPDF2

# Function to extract abstract from PDF
def extract_abstract(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        first_page = pdf_reader.pages[0]
        abstract = first_page.extract_text()
        return abstract
# Function to extract references from PDF
def extract_references(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        last_page = pdf_reader.pages[-1]
        references = last_page.extract_text()
        return references

# File paths for the two PDF documents
document1_filename = "09432648.pdf"
document2_filename = "A_Distributed_AI_ML_Framework_for_D2D_Transmission_Mode_Selection_in_5G_and_Beyond_COMNET_Final.pdf"

# Extract text from the PDF documents
document1_path = document1_filename
document2_path = document2_filename

document1_text = extract_text_from_pdf(document1_path)
document2_text = extract_text_from_pdf(document2_path)
abstract1 = extract_abstract(document1_path)
abstract2 = extract_abstract(document2_path)
references1 = extract_references(document1_path)
references2 = extract_references(document2_path)




# In[22]:


from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from collections import Counter


# Function to find similar words (synonyms) between two documents
def find_similar_words(document1_tokens, document2_tokens):
    similar_words_count = 0

    for word1 in document1_tokens:
        for word2 in document2_tokens:
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            for synset1 in synsets1:
                for synset2 in synsets2:
                    if synset1.wup_similarity(synset2) is not None and synset1.wup_similarity(synset2) > 0.8:
                        similar_words_count += 1

    return similar_words_count




# Tokenize the abstracts
abstract_tokens1 = word_tokenize(abstract1)
abstract_tokens2 = word_tokenize(abstract2)

# Find similar words between the abstracts
similar_words_count = find_similar_words(abstract_tokens1, abstract_tokens2)

# Calculate the total words count
total_words_count = len(abstract_tokens1) + len(abstract_tokens2)

# Calculate the similarity score
similarity_score = similar_words_count / total_words_count

# Determine if the documents are similar or not
if similarity_score >= 0.6:
    print("The documents are similar.")
else:
    print("The documents are not similar.")


# In[15]:


import PyPDF2
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract abstract from PDF
def extract_abstract(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        first_page = pdf_reader.pages[0]
        abstract = first_page.extract_text()
        return abstract

# Function to extract references from PDF
def extract_references(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        last_page = pdf_reader.pages[-1]
        references = last_page.extract_text()
        return references

# File paths for the two PDF documents
document1_filename = "09432648.pdf"
document2_filename = "A_Distributed_AI_ML_Framework_for_D2D_Transmission_Mode_Selection_in_5G_and_Beyond_COMNET_Final.pdf"

# Extract abstracts from the PDF documents
abstract1 = extract_abstract(document1_filename)
abstract2 = extract_abstract(document2_filename)

# Tokenize the abstracts
abstract_tokens1 = word_tokenize(abstract1)
abstract_tokens2 = word_tokenize(abstract2)

# Function to find similar words (synonyms) between two documents
def find_similar_words(document1_tokens, document2_tokens):
    similar_words_count = 0

    for word1 in document1_tokens:
        for word2 in document2_tokens:
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            for synset1 in synsets1:
                for synset2 in synsets2:
                    if synset1.wup_similarity(synset2) is not None and synset1.wup_similarity(synset2) > 0.8:
                        similar_words_count += 1

    return similar_words_count

# Find similar words between the abstracts
similar_words_count = find_similar_words(abstract_tokens1, abstract_tokens2)

# Calculate the total words count
total_words_count = len(abstract_tokens1) + len(abstract_tokens2)

# Calculate the similarity score
similarity_score = similar_words_count / total_words_count

# Determine if the abstracts are similar or not
if similarity_score >= 0.6:
    print("The abstracts are similar.")
else:
    print("The abstracts are not similar.")

# Calculate the semantic similarity using cosine similarity
abstract1_preprocessed = ' '.join(abstract_tokens1)
abstract2_preprocessed = ' '.join(abstract_tokens2)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([abstract1_preprocessed, abstract2_preprocessed])
similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

print(f"Semantic similarity score: {similarity_score}")


# In[7]:





# In[2]:





# In[3]:


get_ipython().system('pip install scienceparse')


# In[1]:


import os
import requests
from io import BytesIO
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel, MarianMTModel, MarianTokenizer
import torch
from scipy.spatial.distance import cosine
import pandas as pd

# Download Science Parse model
def download_science_parse_model():
    model_url = 'https://github.com/allenai/science-parse/releases/download/v2.0.3/scienceparse-v2.0.3-models.tar.gz'
    response = requests.get(model_url)
    tar_file = tarfile.open(fileobj=BytesIO(response.content))
    tar_file.extractall()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize MarianMT model and tokenizer for translation
translator = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
translator_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')

# Function to parse a PDF paper using Science Parse
def parse_paper(paper_path):
    with open(paper_path, 'rb') as f:
        paper_content = f.read()
    parsed_paper = sp.parse_pdf(paper_content)
    return parsed_paper

# Function to translate text from German to English
def translate_text(text):
    # Tokenize input text
    tokens = translator_tokenizer.tokenize(text)
    # Add language prefix to the tokens
    lang_tokens = ['>>de<<'] + tokens + ['>>en<<']
    # Convert tokens to input IDs
    input_ids = translator_tokenizer.convert_tokens_to_ids(lang_tokens)
    # Generate translation
    translation = translator.generate(torch.tensor([input_ids]))
    # Decode translation
    translated_text = translator_tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text

# Calculate semantic similarity using cosine similarity
def calculate_similarity(text1, text2):
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]'])
    outputs = model(torch.tensor([input_ids]))
    embeddings = outputs[0][0]
    text1_embedding = embeddings[1:len(tokens1)+1].mean(dim=0)
    text2_embedding = embeddings[len(tokens1)+2:-1].mean(dim=0)
    similarity = 1 - cosine(text1_embedding.detach().numpy(), text2_embedding.detach().numpy())
    return similarity

# Check if Science Parse model is downloaded, otherwise download it
if not os.path.exists('models/scienceparse-models'):
    download_science_parse_model()

# Initialize Science Parse
sp = scienceparse.ScienceParse(model_dir='models/scienceparse-models')

# Path to train and test data
train_data_path = 'train'
test_data_path = 'test'

# Parse papers using Science Parse
train_data = []
test_data = []

for file in os.listdir(train_data_path):
    if file.endswith('.pdf'):
        paper_path = os.path.join(train_data_path, file)
        parsed_paper = parse_paper(paper_path)
        train_data.append(parsed_paper)

for file in os.listdir(test_data_path):
    if file.endswith('.pdf'):
        paper_path = os.path.join(test_data_path, file)
        parsed_paper = parse_paper(paper_path)
        test_data.append(parsed_paper)

# Calculate semantic similarity for test data
similarities = []

for paper in test_data:
    title1 = paper['title']
    abstract1 = paper['abstract']
    title1_en = translate_text(title1)
    abstract1_en = translate_text(abstract1)

    for train_paper in train_data:
        title2 = train_paper['title']
        abstract2 = train_paper['abstract']
        title_similarity = calculate_similarity(title1_en, title2)
        abstract_similarity = calculate_similarity(abstract1_en, abstract2)
        similarities.append({'Title Similarity': title_similarity, 'Abstract Similarity': abstract_similarity})

# Convert similarities to DataFrame for easier analysis
similarity_df = pd.DataFrame(similarities)

# Calculate average similarity for each paper in test data
similarity_df['Average Similarity'] = similarity_df.mean(axis=1)

# Print the DataFrame
print(similarity_df)


# In[11]:


import os
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
from PyPDF2 import PdfFileReader
from pdfplumber import pdf
from PyPDF2 import PdfFileReader


# In[4]:


get_ipython().system('pip install pdfplumber')


# In[5]:


def download_bert_model():
    tokenizer_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'
    model_url = 'https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin'
    tokenizer_path = 'bert-base-uncased-vocab.txt'
    model_path = 'bert-base-uncased-pytorch_model.bin'

    response = requests.get(tokenizer_url)
    with open(tokenizer_path, 'wb') as f:
        f.write(response.content)

    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)


# In[6]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# In[7]:


def calculate_similarity(text1, text2):
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]'])
    outputs = model(torch.tensor([input_ids]))
    embeddings = outputs[0][0]
    text1_embedding = embeddings[1:len(tokens1)+1].mean(dim=0)
    text2_embedding = embeddings[len(tokens1)+2:-1].mean(dim=0)
    similarity = 1 - cosine(text1_embedding.detach().numpy(), text2_embedding.detach().numpy())
    return similarity


# In[9]:


def read_pdf(paper_path):
    pdf = PdfFileReader(open(paper_path, 'rb'))
    contents = []
    for page_num in range(pdf.numPages):
        page = pdf.getPage(page_num)
        contents.append(page.extract_text())
    return contents


# In[12]:


def translate_text(text):
    translator = GottbertMT(source_lang="de", target_lang="en")
    translation = translator.translate(text)
    return translation


# In[13]:


if not os.path.exists('bert-base-uncased-vocab.txt') or not os.path.exists('bert-base-uncased-pytorch_model.bin'):
    download_bert_model()


# In[ ]:


paper1_path = "path/to/paper1.pdf"
paper2_path = "path/to/paper2.pdf"


# In[16]:


import PyPDF2
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Function to extract abstract from PDF
def extract_abstract(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        first_page = pdf_reader.pages[0]
        abstract = first_page.extract_text()
        return abstract

# Function to extract references from PDF
def extract_references(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        last_page = pdf_reader.pages[-1]
        references = last_page.extract_text()
        return references

# File paths for the two PDF documents
document1_filename = "09432648.pdf"
document2_filename = "A_Distributed_AI_ML_Framework_for_D2D_Transmission_Mode_Selection_in_5G_and_Beyond_COMNET_Final.pdf"

# Extract abstracts from the PDF documents
abstract1 = extract_abstract(document1_filename)
abstract2 = extract_abstract(document2_filename)

# Extract references from the PDF documents
references1 = extract_references(document1_filename)
references2 = extract_references(document2_filename)

# Function to check if the text is in German
def is_german_text(text):
   
    return 'German' in text

# Function to translate German text to English using GottBERT
def translate_german_to_english(text):
    translator = pipeline("translation_en_to_de")
    translation = translator(text, max_length=512)[0]['translation_text']
    return translation

# Check if the documents are in German and translate them to English if necessary
if is_german_text(abstract1):
    abstract1 = translate_german_to_english(abstract1)

if is_german_text(abstract2):
    abstract2 = translate_german_to_english(abstract2)

# Tokenize the abstracts
abstract_tokens1 = word_tokenize(abstract1)
abstract_tokens2 = word_tokenize(abstract2)

# Function to find similar words (synonyms) between two documents
def find_similar_words(document1_tokens, document2_tokens):
    similar_words_count = 0

    for word1 in document1_tokens:
        for word2 in document2_tokens:
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            for synset1 in synsets1:
                for synset2 in synsets2:
                    if synset1.wup_similarity(synset2) is not None and synset1.wup_similarity(synset2) > 0.8:
                        similar_words_count += 1

    return similar_words_count

# Find similar words between the abstracts
similar_words_count = find_similar_words(abstract_tokens1, abstract_tokens2)

# Calculate the total words count
total_words_count = len(abstract_tokens1) + len(abstract_tokens2)

# Calculate the similarity score
similarity_score = similar_words_count / total_words_count

# Determine if the abstracts are similar or not
if similarity_score >= 0.6:
    print("The abstracts are similar.")
else:
    print("The abstracts are not similar.")

# Calculate the semantic similarity using cosine similarity
abstract1_preprocessed = ' '.join(abstract_tokens1)
abstract2_preprocessed = ' '.join(abstract_tokens2)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([abstract1_preprocessed, abstract2_preprocessed])
similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

print(f"Semantic similarity score: {similarity_score}")


# In[19]:


import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from langdetect import detect
import torch
import numpy as np

# Function to extract abstract from PDF
def extract_abstract(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        first_page = pdf_reader.pages[0]
        abstract = first_page.extract_text()
        return abstract

# Function to extract references from PDF
def extract_references(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        last_page = pdf_reader.pages[-1]
        references = last_page.extract_text()
        return references

# File paths for the two PDF documents
document1_filename = "09432648.pdf"
document2_filename = "A_Distributed_AI_ML_Framework_for_D2D_Transmission_Mode_Selection_in_5G_and_Beyond_COMNET_Final.pdf"

# Extract abstracts from the PDF documents
abstract1 = extract_abstract(document1_filename)
abstract2 = extract_abstract(document2_filename)

# Extract references from the PDF documents
references1 = extract_references(document1_filename)
references2 = extract_references(document2_filename)

# Function to check if the text is in German
def is_german_text(text):
    lang = detect(text)
    return lang == 'de'

# Function to translate German text to English using Hugging Face Transformers
def translate_german_to_english(text):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    tokens = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    translated = model.generate(**tokens, max_length=512)
    translated_text = tokenizer.decode(translated[0])
    return translated_text

# Check if the abstracts are in German and translate them to English
if is_german_text(abstract1):
    abstract1 = translate_german_to_english(abstract1)

if is_german_text(abstract2):
    abstract2 = translate_german_to_english(abstract2)

# Check if German and translate them to English if necessary
if is_german_text(references1):
    references1 = translate_german_to_english(references1)

if is_german_text(references2):
    references2 = translate_german_to_english(references2)

# Calculate the similarity score for abstracts using BERT embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

tokens1 = tokenizer(abstract1, truncation=True, padding=True, return_tensors="pt")
tokens2 = tokenizer(abstract2, truncation=True, padding=True, return_tensors="pt")

with torch.no_grad():
    embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1).detach().numpy()

similarity_score_abstracts = cosine_similarity(embeddings1, embeddings2)[0][0]

# Calculate the similarity score for references using BERT embeddings
tokens3 = tokenizer(references1, truncation=True, padding=True, return_tensors="pt")
tokens4 = tokenizer(references2, truncation=True, padding=True, return_tensors="pt")

with torch.no_grad():
    embeddings3 = model(**tokens3).last_hidden_state.mean(dim=1).detach().numpy()
    embeddings4 = model(**tokens4).last_hidden_state.mean(dim=1).detach().numpy()

similarity_score_references = cosine_similarity(embeddings3, embeddings4)[0][0]

print(f"Abstract Similarity Score : {similarity_score_abstracts}")
print(f"References Similarity Score : {similarity_score_references}")


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
similarity_scores = np.array([[similarity_score_abstracts, similarity_score_references]])
sns.heatmap(similarity_scores, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.xticks(np.arange(2) + 0.5, ['Abstracts', 'References'])
plt.yticks([])
plt.xlabel("Document Sections")
plt.title("Document Similarity")
plt.show()


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# Rest of your code...

# Create a graph
G = nx.Graph()

# Add nodes (documents) to the graph
G.add_node(document1_filename)
G.add_node(document2_filename)

# Add edges (similarity scores)
G.add_edge(document1_filename, document2_filename, weight=similarity_score_abstracts)
G.add_edge(document1_filename, document2_filename, weight=similarity_score_references)

# Set positions for the nodes using a force-directed layout
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

# Draw edges
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray')

# Add labels to the nodes
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# Set plot title
plt.title("Document Similarity Visualization")

# Show the plot
plt.axis('off')
plt.show()


# In[3]:





# In[4]:





# In[1]:


import PyPDF2
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def perform_attention(text):
    important_info = text[:500]
    return important_info

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    custom_filters = [lambda x: x.lower(), remove_stopwords]
    preprocessed_text = preprocess_string(text, custom_filters)
    return [word for word in preprocessed_text if word not in stop_words]

def perform_topic_modeling(documents):
    processed_documents = [preprocess_text(doc) for doc in documents]
    dictionary = corpora.Dictionary(processed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]
    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
    return lda_model

# Calculate SIF embeddings
def calculate_sif_embeddings(text, word_embeddings, word_freqs, a=1e-3):
    words = preprocess_text(text)
    word_count = len(words)
    vector = np.zeros(word_embeddings.shape[1])

    for word in words:
        if word in word_embeddings:
            word_weight = a / (a + word_freqs[word] / word_count)
            vector += word_weight * word_embeddings[word]

    if word_count > 0:
        vector /= word_count
    return vector

# Calculate Siamese Network similarity
def calculate_siamese_similarity(embeddings1, embeddings2, embedding3, embedding4):
    class SiameseNetwork(nn.Module):
        def __init__(self, embedding_dim):
            super(SiameseNetwork, self).__init__()
            self.fc = nn.Linear(embedding_dim, 1)

        def forward(self, x1, x2):
            # Cosine
            similarity_score = F.cosine_similarity(x1, x2, dim=1)
            return similarity_score

    return cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1), embeddings3.reshape(1, -1), embeddings4.reshape(1, -1))[0][0]

# Get BERT embeddings
def get_bert_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).detach().numpy()

# Main code
document1_filename = "A_Distributed_AI_ML_Framework_for_D2D_Transmission_Mode_Selection_in_5G_and_Beyond_COMNET_Final.pdf"
document2_filename = "Activating_cavity_by_electrons.pdf"
document3_filename = "Machine_Learning_Methods_Improve_Specificity_in_Ne.pdf"
document4_filename = "09432648.pdf"

text1 = extract_text_from_pdf(document1_filename) 
text2 = extract_text_from_pdf(document2_filename)
text3 = extract_text_from_pdf(document3_filename)
text4 = extract_text_from_pdf(document4_filename)


important_info1 = perform_attention(text1)
important_info2 = perform_attention(text2)
important_info3 = perform_attention(text3)
important_info4 = perform_attention(text4)

lda_model = perform_topic_modeling([text1, text2, text3, text4])

bert_embeddings1 = get_bert_embeddings(important_info1)
bert_embeddings2 = get_bert_embeddings(important_info2)
bert_embeddings1 = get_bert_embeddings(important_info3)
bert_embeddings2 = get_bert_embeddings(important_info4)


sif_embeddings1 = calculate_sif_embeddings(text1, word_embeddings, word_freqs)
sif_embeddings2 = calculate_sif_embeddings(text2, word_embeddings, word_freqs)
sif_embeddings3 = calculate_sif_embeddings(text1, word_embeddings, word_freqs)
sif_embeddings4 = calculate_sif_embeddings(text2, word_embeddings, word_freqs)


siamese_similarity_score = calculate_siamese_similarity(bert_embeddings1, bert_embeddings2,bert_embeddings3, bert_embeddings4)

sif_similarity_score = cosine_similarity(sif_embeddings1.reshape(1, -1), sif_embeddings2.reshape(1, -1))[0][0]

print(f"Siamese Network Similarity Score: {siamese_similarity_score}")
print(f"SIF Similarity Score: {sif_similarity_score}")
print(lda_model.print_topics())


# In[ ]:




