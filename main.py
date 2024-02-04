from dotenv import load_dotenv
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os

def get_vector_store():
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key = os.getenv("QDRANT_API_KEY")
    )

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )

    return vector_store 

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask Qdrant", page_icon="📄")
    st.header("Ask you remote database")

    # create vector store
    vector_store = get_vector_store()

    # create the chain and retriever
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    #show user input 
    user_question = st.text_input("3207-V0005-1101-0001_EN_Installation_Operations_Maintenance_Manual Enter your question here")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()