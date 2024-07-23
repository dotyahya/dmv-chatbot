import os
import pymupdf
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

load_dotenv()

#-------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("ingesting data...")

    # load pdf document
    loader = PyPDFLoader("dataset/dmv_wikipedia.pdf")
    document = loader.load()
    
    # split entire documents into chunks  
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # create vector embeddings and save it in pinecone vector database
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))

# ------------------------------------------------------------------------------------------------

# def extract_text_from_pdf(file_path):
#     """Extract text from PDF and discard images using PyMuPDF"""
#     doc = pymupdf.open(file_path)
#     text = ""
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         text += page.get_text("text")
#     return text

# def extract_text_with_ocr(file_path):
#     """Extract text from PDF using OCR"""
#     images = convert_from_path(file_path)
#     text = ""
#     for image in images:
#         text += pytesseract.image_to_string(image)
#     return text

# def read_txt_path(file_path):
#         with open(file_path, "r", encoding="utf-8") as file:
#             return file.read()

# def process_and_store(file_paths, embeddings, index_name):
#     """Process the files and store the embeddings in Pinecone"""
#     all_texts = []
#     for file_path in file_paths:
#         if file_path.endswith(".pdf"):

#             if file_path == "dataset/file02.pdf":
#                 text = extract_text_with_ocr(file_path)
#             else:
#                 text = extract_text_from_pdf(file_path)

#             # gave AttributeError: 'dict' object has no attribute 'page_content', had to convert text str to langchain Document
#             document = Document(page_content=text, metadata={"source": file_path})
#             text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#             # splitting the document 
#             texts = text_splitter.split_documents([document])
#             all_texts.extend(texts)

#         elif file_path.endswith(".txt"):
#             text = read_txt_path(file_path)
#             # loader = TextLoader(file_path)
#             # document = loader.load()
#             document = Document(page_content=text, metadata={"source": file_path})
#             text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#             texts = text_splitter.split_documents([document])
#             all_texts.extend(texts)

#     print(f"Created {len(all_texts)} chunks from all files.")
#     PineconeVectorStore.from_documents(all_texts, embeddings, index_name=index_name)

# if __name__ == "__main__":
#     print("Ingesting data...")

#     # Initialize OpenAI embeddings
#     embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
#     # File paths
#     file_paths = [
#         "dataset/file02.pdf",
#         "dataset/file03.pdf",
#         "dataset/file04.pdf",
#         "dataset/dmv_wikipedia.txt"
#     ]

#     # Process and store in Pinecone
#     process_and_store(file_paths, embeddings, os.environ.get("INDEX_NAME"))
