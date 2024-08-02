import os
# import time
# import random
import warnings
import streamlit as st
# import threading
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

# storing chat history
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
        # context awareness
        prompt_template = (
            "You are a compassionate and knowledgeable assistant specializing in domestic violence issues. "
            "Answer the following question based on the provided context in a supportive and informative manner.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer in a concise and empathetic manner."
        )

        # merging chat history into context
        context = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history])

        # forming the final prompt
        formatted_prompt = prompt_template.format(context=context, question=prompt)

        response = qa({"question": formatted_prompt, "chat_history": st.session_state.chat_history})
        answer = response.get("answer", "")
        
        # check if the response is empty or indicates no relevant information
        if not answer or "I don't know" in answer:
            answer = "Sorry, no information available on this topic as of yet."
    
    except Exception as e:
        # handle exceptions and provide a fallback message
        answer = "Sorry, the system is currently experiencing high load. Please try again later."
    
    st.session_state.chat_history.append((prompt, answer))

    with st.chat_message("assistant"):
        st.markdown(answer)