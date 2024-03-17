from flask import Flask,render_template,request
from utils import create_text_embedding,search_images,create_indexing_and_load

# from transformers import CLIPModel, CLIPProcessor

app=Flask(__name__)
search_params={"metric_type":"L2",
        "offset":0,
        "ignore_growing":False,
        "params":{"nprobe":8}}
index_params = {
        'metric_type':'L2',
        'index_type':"HNSW",
        'params':{"nlist":256}
    }

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


collection=create_indexing_and_load(index_params)

@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

@app.route('/search',methods=['POST'])
def search():
    text=request.form["query"]
    # text_embedding = create_text_embedding(model,tokenizer,text)
    text_embedding = create_text_embedding(text)
    # text_embedding=text_embedding.numpy()
    # top_images,time_taken=search_images(collection,text_embedding[0],search_params)
    top_images,time_taken=search_images(collection,text_embedding,search_params)
    return  render_template('index.html',top_images=top_images,time_taken=time_taken)
    # return f" Text Embedding is {text_embedding}"
    


app.route('/index',methods=['POST'])
def index():
    return " <H1> This is the index Page</H1> "

if __name__=="__main__":
    app.run(debug=True)
