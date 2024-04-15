#!pip install langchain langchain-mistralai==0.0.4
#!pip install langchain langchain-mistralai==0.0.4

from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from mistralai.client import MistralClient, ChatMessage
from getpass import getpass

# Prompt to put in my API key from Mistral,
api_key= getpass("Enter my Mistral API Key")
client = MistralClient(api_key=api_key)

# Load the knowledge article data (10 KBs about ServiceNow support)
loader = TextLoader("kb_solutions_full.txt")
docs = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Define the embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

# Create the vector store for storing the vectorized documents
vector = FAISS.from_documents(documents, embeddings)

# Define a basic retriever interface
retriever = vector.as_retriever()

# Define the Mistral Model using mostly defaults
model = ChatMistralAI(mistral_api_key=api_key)

# Define a basic prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create a retrieval chain to answer a simple question about a KB
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "What should I do if my upgrade fails on my ServiceNow platform?"})

print(response["answer"])