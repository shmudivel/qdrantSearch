from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import streamlit as st

def get_vector_store():
    # Use Streamlit secrets to access configuration values
    client = qdrant_client.QdrantClient(
        st.secrets["qdrant_config"]["QDRANT_HOST"],
        api_key=st.secrets["qdrant_config"]["QDRANT_API_KEY"]
    )

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=st.secrets["qdrant_config"]["QDRANT_COLLECTION_NAME"],
        embeddings=embeddings,
    )

    return vector_store 

def main():

    st.set_page_config(page_title="Ask Qd")
    st.header("Ask Your Documents")
    st.subheader("Main Features:")
    st.text(" - Ask questions about your documents without needing to re-upload them.")
    st.text(" - Find relevant information from the docs without specific words (technical terms, etc.).")
    st.text(" - Ask complex questions and get answers from different parts of the document.")
    st.subheader("Upcoming Features:")
    st.text(" - Display tables and graphs in the answers.")
    st.text(" - Show the chapter and subchapter of the answers.")
    st.text(" - Include drawings in the answers.")
    st.subheader("Tips:")
    st.text("1. Ask questions related to the content of the document.")
    st.text("2. Questions in natural English are preferred.")
    st.text("3. The more specific your question, the better the answer.")
    st.text("4. If you receive an irrelevant answer, try rephrasing your question.")
    st.text("5. Request chapters and subchapters to double-check the answers.")


    # Create vector store
    vector_store = get_vector_store()

    # Create the chain and retriever
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"]),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Show user input 
    user_question = st.text_input("3207-V0005-1101-0001_EN_Installation_Operations_Maintenance_Manual.pdf")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
