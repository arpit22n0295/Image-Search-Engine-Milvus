import pymilvus
from pymilvus import connections,Collection
import numpy as np
import os
import time 
import torch


def connect_to_milvus():
    uri = 'https://in03-371134389c90924.api.gcp-us-west1.zillizcloud.com'
    token = '7aafbf6c0d6eb7beeaf5d7b23ef0562da520a5da757deb673660b1cd71a07e04475653d333995a6ca4a4ff5509d957130d99e3ef'
    connections.connect(
    alias="default",
    uri=uri,
    token=token)

def create_text_embedding(model,tokenizer,text):

    inputs = tokenizer(text, return_tensors="pt")
    
    # Get the features from CLIP model
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    
    # Normalize features
    embedding = features / features.norm(dim=-1, keepdim=True)

    return embedding


def create_indexing_and_load(index_params):
    connect_to_milvus()
    collection=Collection('Image_Collection')
    collection.create_index(field_name='embedding',index_params=index_params)
    collection.load()
    return collection 

def release_and_drop(collection):
    collection.release()
    collection.drop_index(index_name='embedding')

def search_images(collection,query_embedding,search_params):
    start=time.time()
    results=collection.search(data=[query_embedding],anns_field="embedding",
        param=search_params,
        limit=5,
        expr=None,
        output_fields=['Image_name','embedding'],
        consistency_level='Strong')
    return results[0].ids,time.time()-start

# def get_paths(images_name):
#     paths=[]
#     for image in images_name:
#         paths.append(os.path.join('E:\INTERN\Flask\Local Search Engine\static\dataset\data',image))
#     return paths



    






