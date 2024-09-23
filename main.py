import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.title("CHATBOT: Research Tool 📈")
st.sidebar.title("Article URLs")
main_placeholder = st.empty()
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)
# Remove any empty strings or strings with just spaces
urls = [url for url in urls if url.strip() != ""]

main_placeholder.text(urls)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    main_placeholder.text(data)
    time.sleep(2)
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    # time.sleep(2)
    docs = text_splitter.split_documents(data)
    # main_placeholder.text(docs)
    time.sleep(2)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)
    # create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local("faiss_index")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists("faiss_index"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        # Create the retriever from the vector store
        retriever = new_db.as_retriever(search_type="similarity")
        retrieved_docs = retriever.get_relevant_documents(query)
        result = chain({"input_documents": retrieved_docs, "question": query}, return_only_outputs=True)
        st.subheader("Answer")
        st.write(result["output_text"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
