from langchain.vectorstores import Pinecone
import pinecone

from dotenv import load_dotenv
import os
from src.helper import load_pdf, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
index_name = 'medical-chatbot'

# Load pdf
extracted_data = load_pdf('../data/')

# Create text chunks
text_chunks = text_split(extracted_data)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Iniitializing the Pinecone
pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Creating embeddings from each of the text chunks & storing
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
