from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
from dotenv import load_dotenv
import os
import streamlit as st

def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key = os.getenv("QDRANT_API_KEY")
    )

        # Check connection
    # try:
    #     get_vector_store()
    #     print("Connected to Qdrant server.")
    # except Exception as e:
    #     print("Failed to connect to Qdrant server.")
    #     print(e)

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )

    return vector_store 

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask you remote database")

    # create vector store
    vector_store = get_vector_store()


    # create the chain and retriever
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-3.5-turbo"),
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