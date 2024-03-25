
import json
import networkx as nx
import pickle
import os
from tqdm import tqdm

import pandas as pd

if __name__ == "__main__":
    # Load the JSON data
    print("loading users")
    with open('unique_users.json', encoding='utf-8') as f:
        users = json.load(f)
        
    print("counting users...")
    total_users = 0
    for user in users:
        total_users += 1
        
    print("total users: ", total_users)
    del users
    
    print("loading posts")
    with open('posts.json', encoding='utf-8') as f:
        posts = json.load(f)

    total_threads = 0
    total_comments = 0
    
    for post in posts:
        total_threads += 1
        total_comments += len(post['comments'])
    
    print("total threads: ", total_threads)
    print("total comments: ", total_comments)
    print("total posts: ", total_threads + total_comments)