import streamlit as st
import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS

## Load the Nvidia Api key
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_EMD_KEY'] = os.getenv('NVIDIA_EMD_KEY')


## NVIDIA NIM Inferencing
llm = ChatNVIDIA(
  model="openai/gpt-oss-20b",
  api_key= os.getenv('NVIDIA_API_KEY'), 
  temperature=1,
  top_p=1,
  max_completion_tokens=4096,
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(model="nvidia/llama-3.2-nemoretriever-300m-embed-v1", 
                                                       api_key=os.getenv('NVIDIA_EMD_KEY'),truncate="NONE")
        st.session_state.loader = PyPDFDirectoryLoader('us_census')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_docs= st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


st.title("NVIDIA NIM Demo")
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions: {input}
"""
)

prompt1 = st.text_input("Enter your questions from the documents?")

if st.button("Embed Document"):
    vector_embedding()
    st.write("FAISS vector Store DB is Ready using NvidiaEmbeddings")

if prompt1:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input' : prompt1})
    print("Response Time: ", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        ## Find the Relevant Chunk
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write('-------------------------------------------------------')