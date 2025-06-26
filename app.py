import os
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),  # Use Hugging Face Secret
        model_name="llama3-70b-8192"
    )
    return llm

def create_vector_db():
    loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        raise ValueError("No PDF files found in './data'")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health assistant. Respond thoughtfully to the user's question based on the following context:
{context}
User: {question}
Chatbot:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

print("Initializing Chatbot...")
llm = initialize_llm()
db_path = "./chroma_db"

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

def chatbot_response(user_input):
    if not user_input.strip():
        return "Please provide a valid input."
    return qa_chain.run(user_input)

with gr.Blocks(theme="Respair/Shiki@1.2.1") as app:
    gr.ChatInterface(fn=chatbot_response, title="Mental Health ChatBot")

app.launch()
