from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

from dotenv import load_dotenv
import os

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from src.prompt import *

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
index_name = 'medical-chatbot'

extracted_data = load_pdf('../data/')

text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

pinecone.Pinecone(api_key=PINECONE_API_KEY)

vector_db = Pinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
chain_type_kwargs = {'prompt': PROMPT}

llm = CTransformers(model='../model/llama-2-7b-chat.ggmlv3.q4_0.bin', 
                    model_type='llama',
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vector_db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)