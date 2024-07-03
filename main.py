from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader, UnstructuredURLLoader, WebBaseLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define constants
CHROMA_SETTINGS = {}  # Replace with your actual settings if needed

app = FastAPI()

def get_text_chunks(results, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(results)
    return texts 

def get_vectorstore(results, embeddings_model_name, persist_directory, client_settings, chunk_size, chunk_overlap):
    if embeddings_model_name == "openai":     
        embeddings = OpenAIEmbeddings()
        print('OpenAI embeddings loaded')
    elif embeddings_model_name == "Cohereembeddings":
        embeddings = CohereEmbeddings()
        print('Cohere embeddings loaded')
    else:
        print(f"Embeddings model name provided: {embeddings_model_name}")  # Debug statement
        raise ValueError("Unsupported embeddings model name.")
       
    texts = get_text_chunks(results, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=client_settings)
    db.add_documents(texts)
    return db

def get_conversation_chain(vectorstore, target_source_chunks, model_type):
    retriever = vectorstore.as_retriever(search_kwargs={"k": target_source_chunks})
    
    if model_type == "OpenAI":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    elif model_type == "Llama3":
        llm = Ollama(model="llama3")
    else:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: OpenAI, Llama3")
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def load_input_file(file_path):
    extension = file_path.split(".")[-1]
    first_extension = file_path.split(":")[0]

    if first_extension.lower() in ["http", "https"]:
        if extension == "pdf":
            loader = UnstructuredURLLoader(urls=[file_path])
        else:
            loader = WebBaseLoader(file_path)
    else:
        if extension == "txt":
            loader = TextLoader(file_path=file_path)
        elif extension == "pdf":
            loader = PyMuPDFLoader(file_path=file_path)
        elif extension == "docx":
            loader = Docx2txtLoader(file_path=file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    return loader.load()

@app.post("/process_file/")
async def process_file(file: UploadFile = File(...), model_type: str = Form(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_location = f"{temp_dir}/{file.filename}"
    
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    raw_text = load_input_file(file_location)
    chunk_size = 1000  # Adjust as needed
    chunk_overlap = 200  # Adjust as needed
    embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME')
    persist_directory = os.getenv('PERSIST_DIRECTORY')
    client_settings = CHROMA_SETTINGS  # Replace with actual client settings if needed
    target_source_chunks = 5  # Adjust as needed

    # Debug statements
    print(f"Embeddings model name: {embeddings_model_name}")
    print(f"Persist directory: {persist_directory}")
    print(f"Model type: {model_type}")

    text_chunks = get_text_chunks(results=raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = get_vectorstore(results=text_chunks, embeddings_model_name=embeddings_model_name, persist_directory=persist_directory, client_settings=client_settings, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    global conversation_chain
    conversation_chain = get_conversation_chain(vectorstore=vectorstore, target_source_chunks=target_source_chunks, model_type=model_type)

    return JSONResponse(content={"message": "File processed successfully", "model_type": model_type})

@app.post("/ask_question/")
async def ask_question(question: str = Form(...)):
    global conversation_chain
    response = conversation_chain(question)

    # Debugging the response
    print(f"Response: {response}")

    if isinstance(response, dict) and 'result' in response:
        response_text = response['result']
    else:
        response_text = str(response)
    
    return JSONResponse(content={"response": response_text})

# To run the FastAPI app:
# uvicorn main:app --reload
