from flask import Flask, request, jsonify, render_template
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import os

app = Flask(__name__)

# Initialize session state
session_state = {
    "history": [],  # Keeps track of the chat history
    "generated": ["Hello! tanyakan apapun kepada kami mengenai menstruasi 😇"],  # Default greeting message
    "past": ["Hey! 👋"]  # Initial message
}

def conversation_chat(query, chain, history):
    try:
        result = chain({"question": query, "chat_history": history})
        history.append((query, result["answer"]))
        return result["answer"]
    except Exception as e:
        print(f"Error during conversation: {e}")
        return "Sorry, there was an issue processing your request."

def create_conversational_chain(vector_store):
    try:
        # Create LLM with streaming enabled for faster response
        llm = LlamaCpp(
            streaming=True,  # Enable streaming for faster response
            model_path="model/mistral-7b-instruct-v0.1.Q2_K.gguf",  # Ensure using quantized model
            temperature=0.75,
            top_p=1,
            verbose=True,
            n_ctx=2048  # Reduce n_ctx for faster processing
        )

        # Use limited memory to speed up processing
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=1000)

        # Create conversational chain with reduced 'k' for faster retrieval
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={"k": 1}),  # Reduce k for faster retrieval
            memory=memory
        )
        return chain
    except Exception as e:
        print(f"Error creating conversational chain: {e}")
        return None

def load_and_process_documents():
    dataset_folder = "static/dataset"
    text = []

    try:
        for filename in os.listdir(dataset_folder):
            file_path = os.path.join(dataset_folder, filename)
            if os.path.isfile(file_path) and filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                text.extend(loader.load())
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings for the vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        return create_conversational_chain(vector_store)

    except Exception as e:
        print(f"Error loading and processing documents: {e}")
        return None

# Load and process documents during startup
session_state["chain"] = load_and_process_documents()

# Route to serve the chatbot HTML page
@app.route("/")
def chatbot_page():
    # Kirimkan generated default message ke template
    initial_data = {
        "generated": session_state["generated"],
        "past": session_state["past"]
    }
    return render_template("chatbot_page.html", initial_data=jsonify(initial_data).get_data(as_text=True))

# Route to handle API requests for the chatbot
@app.route("/api/chat", methods=["POST"])
def chatbot_api():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Question is required."}), 400

    if "chain" not in session_state or session_state["chain"] is None:
        return jsonify({"error": "No documents processed or chain not available."}), 400

    chain = session_state["chain"]
    output = conversation_chat(query, chain, session_state['history'])
    session_state['past'].append(query)
    session_state['generated'].append(output)

    return jsonify({"response": output})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)