#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


# In[4]:


# Users wanne User_1, User_2, User_3, User_4
# Features on_scroll, on_focus, on_tick, on_reply
# Item ndio post ambayo tuna recommed kwa User  

# Inabidi data iwe vectorized/ ifanyiwe embeddings 
# THIS IS NOT FINAL


# In[8]:


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# Data hizi Zinatoka kwenye Database (Mfano) the baadae caching mwishonii

data = {
    "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
    "item_id": [101, 102, 101, 103, 102, 104, 103, 104],
    "on_scroll": [1, 0, 2, 0, 1, 0, 3, 2],
    "on_focus": [0, 1, 0, 2, 0, 3, 1, 0],
    "on_tick": [1, 0, 1, 1, 0, 2, 1, 0],
    "on_reply": [0, 1, 0, 0, 2, 1, 0, 3]
}

df = pd.DataFrame(data)

# Sasa, ku recommend inahitajika Uzito au tuseme ile order of importance. Tuseme post the most replies zipewe uzito sana.
# Hapo chini ni hesabu tu za kawaida ya ku generate interaction score.

interaction_weights = {"on_scroll": 1, "on_focus": 2, "on_tick": 3, "on_reply": 4}

df["interaction"] = (
    df["on_scroll"] * interaction_weights["on_scroll"] +
    df["on_focus"] * interaction_weights["on_focus"] +
    df["on_tick"] * interaction_weights["on_tick"] +
    df["on_reply"] * interaction_weights["on_reply"]
)

# Tuzipeleke Kwa Embedding Space/ Vector Space
interaction_matrix = df.pivot(index="user_id", columns="item_id", values="interaction").fillna(0)

# Truncated SVD inachukua items zenye uzito zaidi kwenye embedding space
svd = TruncatedSVD(n_components=2)
user_embeddings = svd.fit_transform(interaction_matrix)

# kNN (User-Based Collaborative Filtering)
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(user_embeddings)

# Similarity Score sasa itafutwe (Wakiwa similar inamaanisha distance kati yao ni ndogo)
user_index = 0  
distances, indices = knn.kneighbors([user_embeddings[user_index]], n_neighbors=3) 
similar_users = indices.flatten()

# Tujaribu kutengeneza a recommendation
recommended_items = df[df["user_id"].isin(similar_users)]["item_id"].unique()
beta_cache = {2: recommended_items}  # Store recommendations for User 1, as a dictionary
print(f"Recommendations for User 1: {beta_cache[2]}")


# In[ ]:




