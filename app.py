from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
GOOGLEAI_API_KEY=os.getenv('GOOGLEAI_API_KEY')
llm= ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLEAI_API_KEY,temperature=0.2,convert_system_message_to_human=True)

loader = PyPDFLoader("documents/Jimit_New_Report.pdf")
doc = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
chunks = text_splitter.split_documents(doc)
'''
# Retrieve embedding function from code env resources
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)
'''

emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder="embedding_model"
)
#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLEAI_API_KEY)


db = Chroma.from_documents(chunks, embedding= embeddings, persist_directory="vector_database")

# Save vector database as persistent files in the output folder
db.persist()

qa_chain=RetrievalQA.from_chain_type(
    llm,
    retriever=db,
    return_source_documents=True
)

question="what is name of professor?"
result=qa_chain({"query":question})
print(result["result"])
