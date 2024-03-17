import milvus
import pymilvus
from pymilvus import db, connections
from milvus import default_server
from pymilvus import Role, Partition
from pymilvus import CollectionSchema,FieldSchema, Collection,DataType,utility
from towhee import ops, pipe, DataCollection
import numpy as np

print("Installed successfully")
p2 = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='text'))
    .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
    .output('text', 'vec')
    )
print(p2('A Black dog').get())