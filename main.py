import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Mental Helth Chatbot - Omdena 🌱")
# btn = st.button("Create Knowledgebase")
# if btn:
#     create_vector_db()

question = st.text_input("Post your thoughts: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])






