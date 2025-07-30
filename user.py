
import boto3
import streamlit as st
import os
import uuid
from uuid import uuid4
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
 
#from langchain.llms.bedrock import Bedrock
from langchain.chat_models import ChatOpenAI


# load s3
s3_client = boto3.client("s3")
bucket_name  = os.getenv("BUCKET_NAME")

#bedrock
#from langchain_community.embeddings import BedrockEmbeddings

#chuck
from langchain.text_splitter import RecursiveCharacterTextSplitter

#pdf loader
from langchain_community.document_loaders import PyPDFLoader

def get_unique_id():
    return  str(uuid.uuid4())

def split_text(pages,chunk_size, chunk_overlap):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs =  text_splitter.split_documents(pages)
    return docs
#bedrock_client = boto3.client("service_name=bedrock-runtime")
#bedrock_client = boto3.client("bedrock-runtime")
#bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")


#bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
openai_embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
def create_vector_store(request_id,splited_docs):
    #vectore_store_faiss = FAISS.from_documents(splited_docs, bedrock_embeddings)
    try:
        vectore_store_faiss = FAISS.from_documents(splited_docs, openai_embeddings )
    except Exception as e:
        st.error(f"Embedding failed: {str(e)}")
        raise

    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectore_store_faiss.save_local(index_name=file_name, folder_path=folder_path)

    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=bucket_name, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=bucket_name, Key="my_pkl.pkl")

    return True
def load_index():
    s3_client.download_file(Bucket=bucket_name, Key="my_faiss.faiss", Filename="/tmp/my_faiss.faiss")
    s3_client.download_file(Bucket=bucket_name, Key="my_faiss.pkl", Filename="/tmp/my_faiss.pkl")

def get_llm():
    return ChatOpenAI(
        model_name="gpt-4",   # or "gpt-4" if you have access
        temperature=0,
    )

def get_response(llm,vectorstore, question ):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']


def main():
    st.header("Subash you are going to rock. Unstoppable")
    load_index()
    dir_list = os.listdir("/tmp/")
    st.write("Files")
    st.write(dir_list)

    faiss_index = FAISS.load_local( index_name="my_faiss", folder_path ="/tmp/", embeddings=openai_embeddings,  allow_dangerous_deserialization=True)
    st.write("INDEX IS READY")
    question = st.text_input("Please ask your question")
    if st.button("Ask Question"):
        with st.spinner("Querying..."):

            llm = get_llm()

            # get_response
            st.write(get_response(llm, faiss_index, question))
            st.success("Done")





if __name__ == "__main__" :
    main()