# main chatbot API file 
# uses FastAPI

# necessary libraries
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from fastapi.middleware.cors import CORSMiddleware

# loading environment variables
load_dotenv()

app = FastAPI()

# adding CORS middleware to allow the React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], # to be updated with the frontend URL for better security
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]         
)

# initializing the embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o-mini", max_tokens=250)
qa = ConversationalRetrievalChain.from_llm(llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever())

# define request and response models
class ChatRequest(BaseModel):
    question: str
    chat_history: list

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = qa({"question": request.question, "chat_history": request.chat_history})
        answer = response.get("answer", "")
        # check if the response is empty or indicates no relevant information
        if not answer or "I don't know" in answer:
            answer = "Sorry, no information available on this topic as of yet."
    except Exception as e:
        # handle exceptions and provide a fallback message
        raise HTTPException(status_code=503, detail="System is currently experiencing high load. Please try again later.")

    return ChatResponse(answer=answer)


# run the program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)