# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
import google.generativeai as genai
import PyPDF2
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from flask_session import Session
from io import BytesIO
from werkzeug.utils import secure_filename
import pandas as pd
from dotenv import load_dotenv

# ================== LOAD ENV ==================
load_dotenv()

# ================== CẤU HÌNH ==================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY không tồn tại!")

genai.configure(api_key=api_key)

GENERATION_MODEL = 'gemini-2.5-flash-lite'
EMBEDDING_MODEL = 'text-embedding-004'

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key")
app.config["SESSION_TYPE"] = "filesystem"
app.config['UPLOAD_FOLDER'] = './static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
Session(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

# ================== KIỂM TRA FILE ==================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ================== RAG DATA ==================
RAG_DATA = {"chunks": [], "embeddings": np.array([]), "is_ready": False}

def extract_pdf_text(pdf_path):
    """Đọc text từ PDF"""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Lỗi PDF {pdf_path}: {e}")
    return text

def create_chunks(directory='./static', size=500):
    """Tạo các đoạn (chunk) từ tất cả PDF trong thư mục"""
    chunks = []
    if not os.path.exists(directory):
        return []
    for f in os.listdir(directory):
        if f.lower().endswith('.pdf'):
            path = os.path.join(directory, f)
            content = extract_pdf_text(path)
            for i in range(0, len(content), size):
                chunk = content[i:i + size].strip()
                if chunk:
                    chunks.append(f"[Nguồn: {f}] {chunk}")
    return chunks

def embed_with_retry(texts, model, retries=5):
    """Tạo embedding có retry nếu lỗi API"""
    embeddings = []
    for text in texts:
        for _ in range(retries):
            try:
                res = genai.embed_content(model=model, content=text)
                embeddings.append(res["embedding"])
                break
            except Exception as e:
                print("Lỗi embedding, thử lại sau 2s:", e)
                time.sleep(2)
        else:
            raise e
    return np.array(embeddings)

def init_rag():
    """Khởi tạo hoặc tải lại RAG"""
    global RAG_DATA
    print("🔄 Đang tải lại RAG...")
    RAG_DATA = {"chunks": [], "embeddings": np.array([]), "is_ready": False}
    chunks = create_chunks()
    if not chunks:
        print("⚠️ Không có PDF hợp lệ trong thư mục static/.")
        return
    try:
        embeddings = embed_with_retry(chunks, EMBEDDING_MODEL)
        RAG_DATA.update({"chunks": chunks, "embeddings": embeddings, "is_ready": True})
        print(f"✅ RAG tải xong: {len(chunks)} đoạn từ {len(os.listdir('./static'))} file PDF.")
    except Exception as e:
        print(f"❌ Lỗi RAG: {e}")
        RAG_DATA["is_ready"] = False

# Tải RAG khi khởi động server
init_rag()

# ================== RAG RETRIEVAL ==================
def retrieve_context(query, k=3):
    """Tìm đoạn liên quan nhất từ RAG"""
    if not RAG_DATA["is_ready"]:
        return "Không có tài liệu."
    try:
        q_vec = embed_with_retry([query], EMBEDDING_MODEL)[0].reshape(1, -1)
        sims = cosine_similarity(q_vec, RAG_DATA["embeddings"])[0]
        idxs = np.argsort(sims)[-k:][::-1]
        return "\n\n---\n\n".join(RAG_DATA["chunks"][i] for i in idxs)
    except Exception as e:
        print("Lỗi retrieve_context:", e)
        return "Lỗi tìm kiếm."

# ================== FORMAT RESPONSE ==================
def format_response(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\s)(.*?)(?<=\S)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'(?m)^\s*\*\s+(.*)', r'• \1', text)
    text = text.replace('\n', '<br>')
    return text

# ================== SESSION HISTORY ==================
def get_ip():
    return request.remote_addr

def get_history():
    key = f"hist_{get_ip()}"
    if key not in session:
        session[key] = []
    return session[key]

def save_history(h):
    key = f"hist_{get_ip()}"
    session[key] = h[-50:]
    session.modified = True

# ================== ROUTES ==================

@app.route('/')
def index():
    status = "Sẵn sàng" if RAG_DATA["is_ready"] else "Chưa có tài liệu"
    return render_template('index.html', rag_status=status)

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '').strip()
    if not msg:
        return jsonify({'response': format_response('Hãy hỏi gì đó nhé!')})

    history = get_history()
    history.append(f"Bạn: {msg}")

    context = retrieve_context(msg)
    recent = "\n".join(history[-10:])

    prompt = f"""
Bạn là AI Thư viện Văn hóa Đọc - THPT Phan Chu Trinh.
Hỗ trợ học sinh tìm tài liệu từ RAG hoặc gợi ý sách/truyện tranh phù hợp.

Tài liệu RAG:
{context}

Lịch sử:
{recent}

Câu hỏi: {msg}

Trả lời:
- Song ngữ (Tiếng Việt trước, sau đó: English Version: ...)
- Dùng <strong>, <em>, • cho danh sách
- Không dùng LaTeX
- Thân thiện, khuyến khích đọc sách
"""

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        res = model.generate_content(prompt)
        ai_text = res.text
        history.append(f"AI: {ai_text}")
        save_history(history)
        return jsonify({'response': format_response(ai_text)})
    except Exception as e:
        print("Lỗi chat:", e)
        return jsonify({'response': format_response('AI đang bận, thử lại sau!')})

# ================== ADMIN ==================
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if (request.form.get('username') == 'buithithuhuong' and
            request.form.get('password') == 'buithithuhuong'):
            session['admin'] = True
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('admin_panel'))
        flash('Sai tài khoản/mật khẩu.', 'error')
    return render_template('admin_login.html')

@app.route('/admin/panel')
def admin_panel():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    histories = []
    for k in session.keys():
        if k.startswith('hist_'):
            ip = k[5:]
            h = session[k]
            if h:
                histories.append({
                    'ip': ip,
                    'messages': len(h),
                    'latest': h[-1],
                    'history': '<br>'.join(h[-10:])
                })

    rag_status = "Sẵn sàng" if RAG_DATA["is_ready"] else "Chưa tải"
    return render_template('admin.html',
                           pdf_files=pdfs,
                           histories=histories,
                           total_users=len(histories),
                           rag_status=rag_status)

@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)
        flash(f'Upload thành công: {file.filename}', 'success')
        init_rag()  # 🔁 Tải lại RAG sau upload
    else:
        flash('Chỉ chấp nhận PDF!', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/delete/<filename>', methods=['POST'])
def admin_delete(filename):
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if os.path.exists(path):
        os.remove(path)
        flash(f'Đã xóa: {filename}', 'success')
        init_rag()  # 🔁 Tải lại RAG sau khi xóa
    return redirect(url_for('admin_panel'))

@app.route('/admin/export_csv')
def export_csv():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    data = []
    for k in session.keys():
        if k.startswith('hist_'):
            ip = k[5:]
            h = session.get(k, [])
            if h:
                data.append({
                    'IP': ip,
                    'Số tin': len(h),
                    'Mới nhất': h[-1],
                    '10 tin cuối': ' | '.join(h[-10:])
                })
    df = pd.DataFrame(data or [{'IP': '-', 'Số tin': 0, 'Mới nhất': '', '10 tin cuối': ''}])
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='lich_su_chat.csv')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Đã đăng xuất.', 'success')
    return redirect(url_for('admin_login'))

# ================== RUN ==================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
