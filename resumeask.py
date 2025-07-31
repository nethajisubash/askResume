import os
import uuid
import boto3
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# Load env vars
load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_DEFAULT_REGION"]  = st.secrets["AWS_DEFAULT_REGION"]
os.environ["HUGGINGFACEHUB_API_TOKEN"]  = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

s3_client = boto3.client("s3")
#bucket_name = os.getenv("BUCKET_NAME")
bucket_name = st.secrets["BUCKET_NAME"]
#openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Embeddings
#openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={'device': 'cpu'}
)


def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def create_vector_store(request_id, splited_docs):
    temp_dir = tempfile.gettempdir()
    file_name = f"{request_id}"
    vectore_store_faiss = FAISS.from_documents(splited_docs, openai_embeddings)
    vectore_store_faiss.save_local(index_name=file_name, folder_path=temp_dir)

    s3_client.upload_file(Filename=os.path.join(temp_dir, file_name + ".faiss"),
                          Bucket=bucket_name, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=os.path.join(temp_dir, file_name + ".pkl"),
                          Bucket=bucket_name, Key="my_faiss.pkl")
    return True


import tempfile
import os

def load_index():
    temp_dir = tempfile.gettempdir()
    faiss_file = os.path.join(temp_dir, "my_faiss.faiss")
    pkl_file = os.path.join(temp_dir, "my_faiss.pkl")

    s3_client.download_file(Bucket=bucket_name, Key="my_faiss.faiss", Filename=faiss_file)
    s3_client.download_file(Bucket=bucket_name, Key="my_faiss.pkl", Filename=pkl_file)

    return FAISS.load_local(index_name="my_faiss",
                            folder_path=temp_dir,
                            embeddings=embeddings,
                            allow_dangerous_deserialization=True)


#def get_llm():
#    return ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

def get_llm():
    return HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, just say that you don't know.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": question})['result']

# Streamlit UI
st.title("RAG My Resume")

if "faiss_index" not in st.session_state:
    with st.spinner("Loading FAISS index from S3..."):
        st.session_state["faiss_index"] = load_index()
        st.success("Index ready!")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar
st.sidebar.title("Model Parameters")
temp = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 1, 4096, 256)
st.sidebar.markdown("Version: 20257312")

# Display past chat
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Subash on his resume"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = get_llm()
        faiss_index = st.session_state["faiss_index"]

        with st.spinner("Searching knowledge base..."):
            response = get_response(llm, faiss_index, prompt)

        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
