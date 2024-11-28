## Closed Domain Question Answering of a Book using Retrieval Augmented Generation (RAG)

# To install the required libraries
!pip install pytorch
!pip install streamlit
!pip install PyPDF2
!pip install langchain
!pip install sentence-transformers
!pip install CTransformers
!pip install chromadb

# To import the required libraries
import torch
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFacePipeline

# Define preprocessing function
def preprocess_text(text):
    text = text.split()
    return ' '.join([str(elem) for elem in text])

# Define setup question-answering pipeline function
def setup_qa_pipeline(pdf_text):
    # To create a document list
    documents = [Document(page_content=pdf_text, metadata={"source": "local"})]
    
    # To split the document into smaller chunks
    # Chunking using Recursive Character Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    book_pdf_splits = text_splitter.split_documents(documents)

    # To create the embeddings of the chunks
    # Using Sentence-Transformers to create the embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # To create a Vector database of the Embeddings
    # Using chromadb as the Vectordb
    vectordb = Chroma.from_documents(documents=book_pdf_splits, embedding=embeddings, persist_directory="chroma_db")

    # To make the vectordb as the retriever
    # To retrieve top 6 documents based on the similarity search
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # To use the pre-trained LLM using Ctransformers
    # LLM used is Mistral 7B
    config = {'max_new_tokens': 4096, 'temperature': 0,'context_length': 4096}
    llm = CTransformers(model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config=config)

    # To create a QA Retrieval Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )

    return qa

# Main Streamlit app code
def main():
    # To give the title of the application
    st.title("PDF Question Answering System using Retrieval Augmented Generation (RAG)")

    # For PDF file processing
    st.header("Upload PDF File")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None:
        # Read PDF file
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        st.subheader("PDF has been uploaded successfully")

        # To Preprocess text and setup QA pipeline
        pdf_text = preprocess_text(pdf_text)
        qa = setup_qa_pipeline(pdf_text)

        # Question answering
        # To take question as the input from user
        st.header("Ask Questions from the PDF")
        question = st.text_input("Ask a question about the PDF content:")
        if st.button("Process and Answer Question"):
            if question:
                response = qa.invoke(question)
                st.subheader("The Answer to the given Question is :")
                st.write("Answer:", response)
                st.write("Thanks for using the Application, have a great day!!!")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
