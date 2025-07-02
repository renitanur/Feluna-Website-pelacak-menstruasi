from flask import Flask, abort, jsonify, render_template, request, redirect, url_for, session, flash
import mysql.connector
from datetime import date
import os
import pathlib
import requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests
from indobert import SentimentAnalyzer
from transformers import pipeline

# Langchain Imports
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import json

app = Flask("Google Login App")
app.secret_key = "your_secret_key"

# Konfigurasi Google OAuth
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
GOOGLE_CLIENT_ID = "69245378667-a6s7plejooi7fto0534bnd9a18ohkb5m.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

def create_flow():
    return Flow.from_client_secrets_file(
        client_secrets_file=client_secrets_file,
        scopes=[
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid"
        ],
        redirect_uri="http://127.0.0.1:5000/callback"
    )

# Konfigurasi Database
db_config = {
    'user': 'root',
    'password': '',  
    'host': 'localhost',
    'database': 'feluna'
}

# Inisialisasi Model Sentimen
model_indobert = 'model'
analyzer_indobert = SentimentAnalyzer(model_indobert)
analyzer = pipeline("sentiment-analysis")

# Inisialisasi LangChain untuk Chatbot
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
app.config['chat_chain'] = load_and_process_documents()

# Decorators for Access Control
def admin_only(f):
    def wrapper(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Anda tidak memiliki akses ke halaman ini.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def user_only(f):
    def wrapper(*args, **kwargs):
        if session.get('role') != 'user':
            flash('Anda tidak memiliki akses ke halaman ini.', 'danger')
            return redirect(url_for('admin'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def login_is_required(f):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            return abort(401)
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# Rute Autentikasi Google
@app.route("/login")
def login():
    flow = create_flow()
    authorization_url, state = flow.authorization_url(prompt='select_account')  # Menambahkan prompt
    session["state"] = state
    return redirect(authorization_url)

# The callback route after Google login
@app.route("/callback")
def callback():
    state = session.get("state")
    if not state:
        flash("State tidak ditemukan dalam sesi. Silakan coba login kembali.", "danger")
        return redirect(url_for('login'))

    flow = create_flow()
    flow.fetch_token(authorization_response=request.url)

    if not flow.credentials:
        flash("Token kredensial tidak ditemukan. Silakan coba login kembali.", "danger")
        return redirect(url_for('login'))

    try:
        id_info = id_token.verify_oauth2_token(
            id_token=flow.credentials.id_token,
            request=google.auth.transport.requests.Request(),
            audience=GOOGLE_CLIENT_ID
        )

        # Extract user details from the ID token
        google_id = id_info.get("sub")
        email = id_info.get("email")
        name = id_info.get("name")

        # Connect to the database using a with block
        with mysql.connector.connect(**db_config) as conn:
            cursor = conn.cursor(dictionary=True)

            # Check if the user exists in the database
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if not user:
                role = 'user'
                cursor.execute(
                    "INSERT INTO users (username, email, role) VALUES (%s, %s, %s)",
                    (name, email, role)
                )
                conn.commit()
                user_id = cursor.lastrowid
            else:
                user_id = user["id"]
                role = user["role"]

            session.update({
                "google_id": google_id,
                "user_id": user_id,
                "name": name,
                "email": email,
                "role": role
            })

    except ValueError as e:
        flash(f"Token verifikasi gagal: {e}", "danger")
        return redirect(url_for('login'))
    except mysql.connector.Error as db_err:
        flash(f"Kesalahan database: {db_err}", "danger")
        return redirect(url_for('login'))
    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return redirect(url_for('login'))

    if role == 'admin':
        return redirect(url_for('admin'))
    elif role == 'user':
        return redirect(url_for('index'))
    else:
        flash("Peran tidak valid. Silakan hubungi dukungan.", "danger")
        return redirect(url_for('login'))

# Rute Utama dan Halaman
@app.route('/')
def beranda():
    if 'user_id' not in session:
        return render_template('beranda.html', is_logged_in=False)
    return render_template('beranda.html', is_logged_in=True)

@app.route('/index')
@user_only
def index():
    return render_template('index.html')

@app.route('/profil')
def profil():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    if session.get('role') == 'admin':
        query = "SELECT * FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        admin = cursor.fetchone()  

        query_users = "SELECT * FROM users"  
        cursor.execute(query_users)
        users = cursor.fetchall() 

        cursor.close()
        conn.close()

        return render_template('admin.html', admin=admin, users=users)  

    else:
        query = "SELECT * FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()  

        cursor.close()
        conn.close()

        return render_template('profil.html', user=user)  

@app.route('/edit_profil', methods=['GET', 'POST'])
def edit_profil():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    user_id = session['user_id']
    is_admin = session.get('role') == 'admin'  

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        birthDate = request.form['birthDate']
        gender = request.form['gender']
        phone = request.form['phone']

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Asumsikan bahwa admin juga disimpan dalam tabel 'users'
        query = """
            UPDATE users
            SET username = %s, email = %s, birthDate = %s, gender = %s, phone = %s
            WHERE id = %s
        """

        cursor.execute(query, (username, email, birthDate, gender, phone, user_id))
        conn.commit()

        cursor.close()
        conn.close()

        flash('Profil berhasil diperbarui.', 'success')

        if is_admin:
            return redirect(url_for('admin'))  
        else:
            return redirect(url_for('profil')) 

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    return render_template('edit_profil.html', user=user, is_admin=is_admin)

# Rute Lainnya
@app.route('/tracking')
def tracking():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))
    today = date.today()
    return render_template('tracking.html', today=today)

@app.route('/treatment')
def treatment():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))
    return render_template('treatment.html')

@app.route('/petunjuk_olahnafas')
def petunjuk_olahnafas():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))
    return render_template('petunjuk_olahnafas.html')

@app.route('/olah_nafas')
def olah_nafas():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))
    return render_template('olah_nafas.html')

@app.route('/peregangan')
def peregangan():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))
    return render_template('peregangan.html')

@app.route('/petunjuk_peregangan')
def petunjuk_peregangan():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))
    return render_template('petunjuk_peregangan.html')

@app.route('/pelatihan')
def pelatihan():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))
    return render_template('pelatihan.html')

# Rute Chatbot
@app.route('/chatbot')
@login_is_required
def chatbot():
    # Ambil histori chat dari session, atau inisialisasi jika belum ada
    chat_history = session.get('chat_history', [])
    generated = session.get('generated', ["Hello! tanyakan apapun kepada kami mengenai menstruasi 😇"])
    past = session.get('past', ["Hey! 👋"])

    initial_data = {
        "generated": generated,
        "past": past
    }

    return render_template("chatbot_page.html", initial_data=json.dumps(initial_data))

@app.route("/api/chat", methods=["POST"])
@login_is_required
def chatbot_api():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Question is required."}), 400

    chain = app.config.get('chat_chain')
    if chain is None:
        return jsonify({"error": "Chatbot is not available."}), 500

    try:
        result = chain({"question": query, "chat_history": session.get('chat_history', [])})
        answer = result.get("answer", "Maaf, saya tidak mengerti pertanyaan Anda.")
        
        # Update histori chat dalam session
        chat_history = session.get('chat_history', [])
        chat_history.append((query, answer))
        session['chat_history'] = chat_history

        # Update histori yang dikirim ke front-end
        past = session.get('past', [])
        past.append(query)
        session['past'] = past

        generated = session.get('generated', [])
        generated.append(answer)
        session['generated'] = generated

    except Exception as e:
        print(f"Error during conversation: {e}")
        answer = "Maaf, terjadi kesalahan saat memproses permintaan Anda."

    return jsonify({"response": answer})

# Fungsi Sentimen
def get_sentiment(text):
    try:
        # Analisis dengan indobert
        result = analyzer_indobert(text)
        print(f"Model output for text '{text}': {result}")  # Debugging

        if result:
            sentiment_label = result[0]["label"].lower()
            sentiment_score = result[0]["score"]

            # Logika untuk klasifikasi sentimen
            if sentiment_label == "positive":
                if sentiment_score >= 0.6:
                    return "Positif"
                else:
                    return "Netral"
            elif sentiment_label == "negative":
                if sentiment_score >= 0.6:
                    return "Negatif"
                else:
                    return "Netral"
            else:
                return "Netral"
        else:
            return "Netral"
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return "Netral"

# Rute Feedback
# Rute Feedback
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Mendapatkan feedback dari form
        feedback_content = request.form.get('feedback')
        user_id = session.get('user_id')

        if not feedback_content:
            flash('Feedback tidak boleh kosong.', 'danger')
            return redirect(url_for('feedback'))

        # Insert feedback ke database
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            sentiment = get_sentiment(feedback_content)
            query = "INSERT INTO feedback (user_id, content, sentiment) VALUES (%s, %s, %s)"
            cursor.execute(query, (user_id, feedback_content, sentiment))
            conn.commit()
            flash('Terima kasih atas feedback Anda!', 'success')
        except mysql.connector.Error as err:
            flash(f"Terjadi kesalahan saat menyimpan feedback: {err}", 'danger')
        finally:
            cursor.close()
            conn.close()

        return redirect(url_for('feedback'))

    # Mengambil semua feedback untuk ditampilkan
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT f.content, f.created_at, u.username, f.sentiment
            FROM feedback f
            JOIN users u ON f.user_id = u.id
            ORDER BY f.id DESC
        """
        cursor.execute(query)
        reviews = cursor.fetchall()
    except mysql.connector.Error as err:
        flash(f"Terjadi kesalahan saat mengambil data feedback: {err}", "danger")
        reviews = []
    finally:
        cursor.close()
        conn.close()

    return render_template('feedback.html', reviews=reviews)

@app.route('/data', methods=['GET'])
def send_data():
    data = {
        "message": "Hello feluna",
        "status": "success"
    }
    return jsonify(data), 200

# Endpoint untuk menerima data dari peregangan
@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    return jsonify({"message": "Data received successfully", "received_data": data}), 200

# Rute Admin
@app.route('/admin')
@admin_only
def admin():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:
        # Mengambil data pengguna
        cursor.execute("SELECT id, username, email, role FROM users")
        users = cursor.fetchall()

        # Mengambil data feedback
        cursor.execute("""
            SELECT f.content, f.created_at, u.username, f.sentiment 
            FROM feedback f 
            JOIN users u ON f.user_id = u.id 
            ORDER BY f.id DESC
        """)
        feedbacks = cursor.fetchall()

        # Memisahkan feedback berdasarkan sentimen
        positive_feedbacks = [f for f in feedbacks if f['sentiment'] == 'Positif']
        negative_feedbacks = [f for f in feedbacks if f['sentiment'] == 'Negatif']
        neutral_feedbacks = [f for f in feedbacks if f['sentiment'] == 'Netral']

        sentiment_distribution = {
            "positive": len(positive_feedbacks),
            "negative": len(negative_feedbacks),
            "neutral": len(neutral_feedbacks),
        }

        # Ambil informasi admin dari session
        name = session.get("name", "Admin")
        email = session.get("email", "admin@example.com")

    except mysql.connector.Error as err:
        flash(f"Terjadi kesalahan saat mengambil data: {err}", "danger")
        users, feedbacks = [], []
        positive_feedbacks, negative_feedbacks, neutral_feedbacks = [], [], []
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
    finally:
        cursor.close()
        conn.close()

    return render_template(
        'admin.html',
        name=name,
        email=email,
        users=users,
        feedbacks=feedbacks,
        positive_feedbacks=positive_feedbacks,
        negative_feedbacks=negative_feedbacks,
        neutral_feedbacks=neutral_feedbacks,
        sentiment_distribution=sentiment_distribution
    )


def get_reviews_from_db():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, content FROM feedback")
        reviews = cursor.fetchall()
        return reviews
    except mysql.connector.Error as err:
        flash(f"Terjadi kesalahan saat mengambil ulasan: {err}", "danger")
        return []
    finally:
        cursor.close()
        conn.close()

def get_admin_details(admin_id):
    """Fungsi untuk mendapatkan detail admin berdasarkan admin_id."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT username, email, birthDate, gender, phone FROM users WHERE id = %s", (admin_id,))
    admin = cursor.fetchone()  
    cursor.close()
    conn.close()
    return admin

def get_all_users():
    """Fungsi untuk mendapatkan daftar semua pengguna dari database."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username, email FROM users")  
    users = cursor.fetchall()  
    cursor.close()
    conn.close()
    return users

# Rute Logout
@app.route('/logout')
def logout():
    session.clear()  # Membersihkan seluruh sesi
    flash('Anda telah logout.', 'success')
    return redirect(url_for('beranda'))

if __name__ == '__main__':
    app.run(port=5000, debug=True) 
