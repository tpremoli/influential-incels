import json
import networkx as nx
import pickle
import os
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import pipeline

EMOTIONS = ["admiration", "amusement", "approval", "caring", "anger", "annoyance",
            "disappointment", "disapproval", "confusion", "desire", "excitement",
            "gratitude", "joy", "disgust", "embarrassment", "fear", "grief",
            "curiosity", "love", "optimism", "pride", "relief", "nervousness",
            "remorse", "sadness", "realization", "surprise", "neutral"]


def create_graph(users, posts):
    # Initialize a directed graph
    incel_graph = nx.DiGraph()

    # Add nodes (users) to the graph
    print("adding users (nodes) to graph")
    for user in tqdm(users, desc="Processing users"):
        incel_graph.add_node(user['user_id'], username=user['username'])

    # Define weights for different types of interactions
    mention_quote_weight = 1.0
    general_comment_weight = 0.5

    # Add edges (interactions) to the graph
    print("Adding edges (interactions) to graph...")
    for post in tqdm(posts, desc="Processing posts"):
        for comment in post['comments']:
            # Direct interaction through mentions or quotes
            has_direct_interaction = False

            # If the comment mentions other users
            if comment['mentioned_users']:
                has_direct_interaction = True
                for mentioned_user in comment['mentioned_users']:
                    incel_graph.add_edge(comment['user_id'], mentioned_user, weight=mention_quote_weight)

            # If the comment quotes other posts
            if comment['quoted_posts']:
                has_direct_interaction = True
                for quoted_post in comment['quoted_posts']:
                    # Find the original poster and add an edge
                    original_poster = next((item['user_id'] for item in posts if item['post_id'] == quoted_post), None)
                    if original_poster:
                        incel_graph.add_edge(comment['user_id'], original_poster, weight=mention_quote_weight)

            # General comment interaction with the original post's author
            if not has_direct_interaction and comment['user_id'] != post['user_id']:
                incel_graph.add_edge(comment['user_id'], post['user_id'], weight=general_comment_weight)

    print("graph calculation complete. Saving to file")
    with open('forum_graph.pkl', 'wb') as f:
        pickle.dump(incel_graph, f)
    
    return incel_graph

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

def split_text(text, max_length):
    # Tokenize the text and get the tokens
    tokens = tokenizer.tokenize(text)
    
    # Initialize chunks
    chunks = []

    for i in range(0, len(tokens), max_length - 2):  # Adjust for special tokens
        # Convert tokens back to text
        chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_length - 2])
        chunks.append(chunk)
    
    return chunks

def get_weighted_mean_emotion_score(chunks, model):
    total_length = sum(len(chunk) for chunk in chunks)
    
    # Initialize a dictionary for emotion labels with initial score of 0
    weighted_scores = {emotion: 0 for emotion in EMOTIONS}

    for chunk in chunks:
        chunk_scores = model(chunk)  # This returns a list of dictionaries for each chunk
        weight = len(tokenizer.tokenize(chunk)) / total_length  # Update the weight calculation based on tokens

        for score_dict in chunk_scores[0]:  # chunk_scores[0] contains our list of dictionaries
            emotion = score_dict['label']
            score = score_dict['score']
            if emotion in weighted_scores:
                weighted_scores[emotion] += score * weight

    return weighted_scores

def calc_emotions(posts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    goemotions = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)

    rows = []

    for post in tqdm(posts, desc="Processing posts"):
        # Split and analyze the post's text sentiment
        post_chunks = split_text(post['text_content'], 512)
        post_mean_score = get_weighted_mean_emotion_score(post_chunks, goemotions)
        
        # Calculate post length in words
        post_length = len(post['text_content'].split())
        
        # Ensure the order of scores matches the order of emotions
        post_scores = [post_mean_score[emotion] for emotion in EMOTIONS]
        rows.append([post['post_id'], post['user_id'], post_length] + post_scores)

        # Analyze sentiments in the comments
        for comment in post['comments']:
            comment_chunks = split_text(comment['text_content'], 512)
            comment_mean_score = get_weighted_mean_emotion_score(comment_chunks, goemotions)

            # Calculate comment length in words
            comment_length = len(comment['text_content'].split())

            # Ensure the order of scores matches the order of emotions
            comment_scores = [comment_mean_score[emotion] for emotion in EMOTIONS]
            rows.append([comment['post_id'], comment['user_id'], comment_length] + comment_scores)

    # Save to CSV with post length column
    columns = ['post_id', 'user_id', 'post_length'] + [emotion for emotion in EMOTIONS]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('sentiment_analysis_goemotions.csv', index=False)

def calc_sentiments(posts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cardiffRoberta = pipeline(task="text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None, device=device)

    rows = []

    for post in tqdm(posts, desc="Processing posts"):
        # Split and analyze the post's text sentiment
        post_chunks = split_text(post['text_content'], 512)
        post_sentiment = get_weighted_mean_emotion_score(post_chunks, cardiffRoberta)
        
        # Calculate post length in words
        post_length = len(post['text_content'].split())
        
        # Ensure the order of scores matches the sentiment categories
        post_scores = [post_sentiment.get(sentiment, 0) for sentiment in ['negative', 'neutral', 'positive']]
        rows.append([post['post_id'], post['user_id'], post_length] + post_scores)

        # Analyze sentiments in the comments
        for comment in post['comments']:
            comment_chunks = split_text(comment['text_content'], 512)
            comment_sentiment = get_weighted_mean_emotion_score(comment_chunks, cardiffRoberta)

            # Calculate comment length in words
            comment_length = len(comment['text_content'].split())

            # Ensure the order of scores matches the sentiment categories
            comment_scores = [comment_sentiment.get(sentiment, 0) for sentiment in ['negative', 'neutral', 'positive']]
            rows.append([comment['post_id'], comment['user_id'], comment_length] + comment_scores)

    # Save to CSV with post length column
    columns = ['post_id', 'user_id', 'post_length', 'negative', 'neutral', 'positive']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('sentiment_analysis_cardiff.csv', index=False)

if __name__ == "__main__":
    # Load the JSON data
    print("loading users")
    with open('unique_users.json', encoding='utf-8') as f:
        users = json.load(f)

    print("loading posts")
    with open('posts.json', encoding='utf-8') as f:
        posts = json.load(f)
    
    # check if forum_graph.pkl exists, if so load it, else calculate it
    if os.path.exists('forum_graph.pkl'):
        print("Loading graph from file...")
        with open('forum_graph.pkl', 'rb') as f:
            incel_graph = pickle.load(f)
    else:
        print("Calculating graph...")
        incel_graph = create_graph(users,posts)
                
    print("calculating sentiments")
    calc_sentiments(posts)