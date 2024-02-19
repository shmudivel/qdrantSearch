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
    st.header("Ask your documents")
    st.subheader("Main features:")
    st.text(" - Ask questions about your douments without need to reupload them")
    st.text(" - Find the relavant info from the docs without specific words (technical terms, etc.)")
    st.text(" - Ask complex questions and get the answer from different parts of the document")
    st.subheader("Upcoming features:")
    st.text("Show the table and graphs in the answer")
    st.text("Show the chapter and subchapter of the answer")
    st.text("Show the drowings in the answer")
    st.subheader("Tips:")
    st.text("1. Ask questions related to the content of the doc")
    st.text("2. Ask questions in natural language English would be better")
    st.text("3. The more specific your question the better the answer")
    st.text("4. If you get an answer that is not relevant, try rephrasing your question")
    st.text("5. Ask for chapter and subchapter to double check the answer")

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
