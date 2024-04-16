'''
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
LANGCHAIN_API_KEY=os.getenv('GOOGLEAI_API_KEY')

loader = PyPDFLoader("documents/Jimit_New_Report.pdf")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder="embedding_model"
)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
print(retriever.get_relevant_documents("what is name of professor")[0])
'''

from fastapi import FastAPI, Request
#from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities.python import PythonREPL
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from pydantic import BaseModel

load_dotenv()
GOOGLEAI_API_KEY=os.getenv('GOOGLEAI_API_KEY')


loader = PyPDFLoader("documents/Flight_Schedule_and_Routes.pdf")
data = loader.load()

class Request(BaseModel):
    query: str

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)


emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder="embedding_model"
)
vector_store = FAISS.from_documents(texts, embeddings)


def search_knowledge_base(query):
    docs = vector_store.similarity_search(query, k=3)
    return [doc.page_content for doc in docs]


def dummy_api(query):
  
    return {
        "flight_number": "ABC123",
        "origin": "New York",
        "destination": "Los Angeles",
        "departure_time": "2023-04-04 09:00:00",
        "arrival_time": "2023-04-04 11:30:00",
    }


search_knowledge_base_tool = Tool(name="Search Knowledge Base", func=search_knowledge_base, description="Search the knowledge base for relevant flight information")
dummy_api_tool = Tool(name="Dummy API", func=dummy_api, description="Call the dummy API to get additional flight information")



LLM= ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLEAI_API_KEY,temperature=0.2,convert_system_message_to_human=True)
tools = [search_knowledge_base_tool, dummy_api_tool]
agent = initialize_agent(tools, LLM, agent="zero-shot-react-description", verbose=True)


app = FastAPI()

@app.post("/query")
async def query_handler(request: Request):
    query = await request.body()
    response = agent.run(query)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)