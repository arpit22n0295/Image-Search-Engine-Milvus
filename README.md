# Image-Search-Engine-Milvus

Introducing a Text-based Image Search Engine:

Our Text-based Image Search Engine offers a sophisticated solution for finding similar images within a vast database containing thousands or even millions of images. We leverage the powerful functionality of the Milvus Vector Database to efficiently store embeddings of all images in the database. These embeddings are generated using the CLIP model, which extracts embeddings for both images and text.

Key Features:

Utilizes the Milvus Vector Database: Our engine capitalizes on the robust capabilities of the Milvus Vector Database to store and efficiently retrieve image embeddings.
CLIP Model Integration: We employ the CLIP model to calculate embeddings for both images and text, facilitating effective cross-modal search capabilities.
Flask Web Application: We've built a user-friendly web application using Flask, enabling users to interact with the search engine seamlessly.
Scalable Deployment on AWS EC2: Our solution is designed for scalability, and we've deployed it on AWS EC2 instances to ensure reliability and performance.
How It Works:

Embedding Generation: Images in the database are processed to generate embeddings using the CLIP model, capturing both visual and semantic information.
Indexing with Milvus: The embeddings are then efficiently indexed and stored in the Milvus Vector Database for fast retrieval.
Text-based Search: Users can input text queries, which are converted into embeddings using the CLIP model. The engine then retrieves images with embeddings most similar to the query.
Web Interface: The Flask-based web application provides a user-friendly interface for querying and browsing similar images.
By combining state-of-the-art technologies like the CLIP model and the Milvus Vector Database, our Text-based Image Search Engine offers a powerful solution for discovering visually and semantically similar images within large-scale image collections.
