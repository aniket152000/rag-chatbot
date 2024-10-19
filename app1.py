import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
#Python 3.10.9

# Load environment variables
load_dotenv()

# Set title of the application
st.title("RAG-Based Application Using Gemini Model")

# Load PDF document
loader = PyPDFLoader("Company-Policies-Manuals.pdf")
data = loader.load()

# Split the loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create a directory for Chroma DB
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="./chroma_db"  # Ensure a valid persist directory   
)

# Set up the retriever with similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize the language model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
)

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input for queries
query = st.chat_input("Say something: ") 

if query:
    # Store the user's query in history
    st.session_state.history.append({"user": query})

    # System prompt for the model
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Set up the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get the response from the RAG chain
    response = rag_chain.invoke({"input": query})

    # Store the assistant's response in history
    st.session_state.history.append({"assistant": response['answer']})


# Display the chat history
if st.session_state.history:
    for chat in st.session_state.history:
        if "user" in chat:
            st.markdown(f"<div style='text-align: right; color: black;'><b>You:</b> {chat['user']}</div><br> ", unsafe_allow_html=True)
        if "assistant" in chat:
            st.markdown(f"<div style='text-align: left; color: black;'><b>Assistant:</b> {chat['assistant']}</div><br> ", unsafe_allow_html=True)