from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader, WebBaseLoader   
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from typing import Dict
import os
import bs4
import requests
from dotenv import find_dotenv, load_dotenv

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("loaded model embedding")

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_api)

print("llm loaded")

app = FastAPI()
dbs: Dict[str, FAISS] = {}
prompts = {}

# Model for URL processing API
class URLRequest(BaseModel):
    url: str

# Model for chat API
class ChatRequest(BaseModel):
    chat_id: str
    question: str
    

# # Process Web URL API
@app.post("/process_url")
async def process_url(request: URLRequest):
    try:
        # Load web content using WebBaseLoader
        loader = WebBaseLoader(
            web_paths=(request.url,),
            )
        docs = loader.load()

        print("yes")

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        print(chunks)

        # Generate a unique chat_id and store the chunks in FAISS
        chat_id = f"url_{len(request.url)}"
        dbs[chat_id] = FAISS.from_documents(chunks, embeddings)

        # Save the FAISS index locally
        dbs[chat_id].save_local(f"faiss_db/faiss_db_{chat_id}")

        return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

# Process PDF Document API
@app.post("/process_pdf")
async def process_pdf(user_id: str, file: UploadFile = File(...)):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        # Save the uploaded file
        file_path = f"files/file_{user_id}.pdf"
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Process and index the PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        chunks = pages
        print(chunks)
        db_name = f"db_{user_id}"
        print(db_name)
        dbs[db_name] = FAISS.from_documents(chunks, embeddings)

        print("first run")
        dbs[db_name].save_local(f"faiss_db/faiss_{db_name}")
        print("ran successfully")
        
        # Remove the local file after processing
        os.remove(file_path)
        
        return {"chat_id": db_name, "message": "PDF content processed and stored successfully."}
    
    except requests.exceptions.RequestException as err:
        return JSONResponse(status_code=400, content={"error": f"Error occurred during file upload: {str(err)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.chat_id
    query = request.question
    try:
        db_name = f"faiss_db_{user_id}"
        try:
            new_db = FAISS.load_local(f"faiss_db/{db_name}", embeddings, allow_dangerous_deserialization=True)
        except FileNotFoundError as file_err:
            return JSONResponse(status_code=404, content={'error': f"File not found. For user {user_id}, first upload the docs: {str(file_err)}"})

        # Embed the query using HuggingFaceEmbeddings
        query_embedding = embeddings.embed_query(query)

        # Similarity search by vector
        nearest_docs = new_db.similarity_search_by_vector(query_embedding, k=5)  # Retrieve 5 nearest documents
        print(nearest_docs)

        # text = ""
        # for i in nearest_docs:
        #     text = text + i.page_content
        # final_query = query + text

        # result = llm.invoke(query)
        def format_docs(docs_data):
            return "\n\n".join(doc.page_content for doc in docs_data)
        
        prompt = hub.pull("rlm/rag-prompt")

        retriever = new_db.as_retriever(search_kwargs={'k': 3})
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(query)

        return {"Answer":result}

    except requests.exceptions.HTTPError as err:
        return JSONResponse(status_code=404, content={'error': str(err)})

    except Exception as e:
        return JSONResponse(status_code=500, content={'error': f"Unexpected error: {str(e)}"})
