import json
import networkx as nx
from tqdm import tqdm

# Load the JSON data
print("loading users")
with open('unique_users.json', encoding='utf-8') as f:
    users = json.load(f)

print("loading posts")
with open('posts.json', encoding='utf-8') as f:
    posts = json.load(f)

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes (users) to the graph
print("adding users (nodes) to graph")
for user in tqdm(users, desc="Processing users"):
    G.add_node(user['user_id'], username=user['username'])

# Add edges (interactions) to the graph
print("adding edges (interactions) to graph")
for post in tqdm(posts, desc="Processing posts"):
    # Add an edge for the post itself (if the author is replying to someone)
    if post['mentioned_users']:
        for mentioned_user in post['mentioned_users']:
            G.add_edge(post['user_id'], mentioned_user)

    # Add edges for each comment
    for comment in post['comments']:
        # If the comment mentions other users
        if comment['mentioned_users']:
            for mentioned_user in comment['mentioned_users']:
                G.add_edge(comment['user_id'], mentioned_user)

        # If the comment quotes other posts
        for quoted_post in comment['quoted_posts']:
            # Find the original poster and add an edge
            original_poster = next((item['user_id'] for item in posts if item['post_id'] == quoted_post), None)
            if original_poster:
                G.add_edge(comment['user_id'], original_poster)

# Apply the PageRank algorithm
print("applying pagerank algorithm to graph")
pagerank = nx.pagerank(G)

# Sort users by their PageRank scores
print("sorting users by pagerank scores")
sorted_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

# Display the top 10 influential users
print("Analysis complete. Top 10 users:")
for user_id, score in sorted_users[:10]:
    print(f"User ID: {user_id}, Score: {score}")
