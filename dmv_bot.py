import os
import time
import random
import warnings
import streamlit as st
import threading
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

warnings.filterwarnings("ignore")

load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o-mini", max_tokens=250)
qa = ConversationalRetrievalChain.from_llm(llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever())

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # display chat messages from history on app rerun
# st.title("Domestic Violence One-Stop")
# for question, answer in st.session_state.chat_history:
#     with st.chat_message("user"):
#         st.markdown(question)
#     with st.chat_message("assistant"):
#         st.markdown(answer)

# # accept user input
# if prompt := st.chat_input("Ask me anything about domestic violence"):
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     response = qa({"question": prompt, "chat_history": st.session_state.chat_history})
#     answer = response["answer"]

#     st.session_state.chat_history.append((prompt, answer))

#     with st.chat_message("assistant"):
#         st.markdown(answer)

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# handle responses with a timeout
# def get_response_with_timeout(prompt, chat_history):
#     # function to get the response from the QA model
#     def get_response():
#         nonlocal response
#         try:
#             response = qa({"question": prompt, "chat_history": chat_history})
#         except Exception as e:
#             response = {"answer": "An error occurred while processing your request."}

#     response = None
#     thread = threading.Thread(target=get_response)
#     thread.start()
#     thread.join(timeout=20) # timeout in seconds

#     if thread.is_alive():
#         return "The system is currently experiencing high load. Please try again later."
#     # return response.get("answer", "Sorry, no information available on this topic as of yet.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# display chat messages from history on app rerun
st.title("Domestic Violence One-Stop")
for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)

# accept user input
if prompt := st.chat_input("Ask me anything about domestic violence"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        response = qa({"question": prompt, "chat_history": st.session_state.chat_history})
        answer = response.get("answer", "")
        
        # Check if the response is empty or indicates no relevant information
        if not answer or "I don't know" in answer:
            answer = "Sorry, no information available on this topic as of yet."
    
    except Exception as e:
        # handle exceptions and provide a fallback message
        answer = "Sorry, the system is currently experiencing high load. Please try again later."
    
    st.session_state.chat_history.append((prompt, answer))

    with st.chat_message("assistant"):
        st.markdown(answer)