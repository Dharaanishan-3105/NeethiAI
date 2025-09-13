from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from authlib.integrations.flask_client import OAuth
from datetime import datetime
import os
import json
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from urllib.parse import urlparse
import google.generativeai as genai
import re
import random
from time import sleep
import langdetect
from langdetect import detect
import PyPDF2
import docx
import easyocr
from PIL import Image, ImageFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Pillow compatibility shim for deprecated resampling constants used by dependencies (e.g., EasyOCR)
try:
    # Pillow >= 10 removed these aliases; map to Resampling equivalents
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    if not hasattr(Image, 'BILINEAR'):
        Image.BILINEAR = Image.Resampling.BILINEAR
    if not hasattr(Image, 'BICUBIC'):
        Image.BICUBIC = Image.Resampling.BICUBIC
except Exception:
    pass
try:
    # Allow loading slightly corrupted/truncated images rather than erroring out
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass
try:
    import pytesseract
    _has_tesseract = True
except Exception:
    _has_tesseract = False
import cv2
import numpy as np
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Database configuration - use Render's DATABASE_URL if available
database_url = os.getenv('DATABASE_URL')
if database_url:
    # Render provides DATABASE_URL, use it directly
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print(f"🔗 Using production database: {database_url[:50]}...")
    print("✅ Database URL found - ready for production!")
else:
    # Local development fallback
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://neethi_user:HariDharaan%402025@localhost/neethi_ai'
    print("🔗 Using local development database")
    print("⚠️  WARNING: DATABASE_URL not found! Make sure to set it in Render environment variables.")
    print("💡 To fix: Link your PostgreSQL database to the web service in Render dashboard")
    print("📋 Environment variables available:", [k for k in os.environ.keys() if 'DATABASE' in k.upper()])

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Limit uploads to 10 MB to prevent crashes
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 10 * 1024 * 1024))

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID', 'your-google-client-id'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET', 'your-google-client-secret'),
    server_metadata_url='https://accounts.google.com/.well-known/openid_configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)  # Nullable for OAuth users
    name = db.Column(db.String(100), nullable=False)
    google_id = db.Column(db.String(100), unique=True, nullable=True)
    profile_picture = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    auth_provider = db.Column(db.String(20), default='email')  # 'email' or 'google'
    
    # Relationship to chats
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text)  # JSON string of sources
    language = db.Column(db.String(10), default='en')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feature_used = db.Column(db.String(50))  # Which feature was used

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Make config available in templates
@app.context_processor
def inject_config():
    return dict(config=app.config)

# Initialize EasyOCR reader with Tamil support
# EasyOCR reader will be initialized when needed to avoid startup issues
reader = None

# Language Detection and Support
def detect_language(text):
    """Detect if text is in Tamil or English"""
    try:
        lang = detect(text)
        return 'ta' if lang == 'ta' else 'en'
    except:
        return 'en'

def get_language_prompt(language):
    """Get language-specific prompts"""
    if language == 'ta':
        return {
            'system': "நீங்கள் ஒரு சட்ட AI உதவியாளர். இந்திய குடிமக்களுக்கு அவர்களின் உரிமைகளை புரிந்துகொள்ள உதவுங்கள்.",
            'response_instruction': "தமிழில் பதிலளிக்கவும் மற்றும் சட்ட பிரிவுகளை குறிப்பிடவும்."
        }
    else:
        return {
            'system': "You are a legal AI assistant helping citizens understand their rights.",
            'response_instruction': "Answer in English and mention relevant legal sections."
        }

# Configure Gemini API
def configure_gemini_api():
    api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyDBHb0TxrV7nrIZ3bAgi1YCWrMoLPBQPq8')
    try:
        if not api_key:
            app.logger.error("GEMINI_API_KEY not found in environment variables")
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        app.logger.info("Gemini API configured successfully")
        return model
    except Exception as e:
        app.logger.error(f"Error configuring Gemini API: {str(e)}")
        return None

# Initialize Gemini model
model = configure_gemini_api()

# Allow loading truncated images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Friendly error for oversized uploads
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'error': 'File too large. Please upload an image under 10 MB.'}), 413

# Translation helper
def translate_text(text, target_lang, model):
    try:
        if target_lang not in ['en', 'ta'] or not text.strip():
            return text
        prompt = f"""Translate the following text to {'English' if target_lang=='en' else 'Tamil'} preserving legal meaning and terminology:

Text:
{text}
"""
        response = model.generate_content(prompt)
        return response.text if response and response.text else text
    except Exception:
        return text

# PDF Parsing
def extract_text_from_pdf(file):
    full_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text() or ""
            full_text += text
        return full_text[:4000]  # Gemini safe limit
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

# Word Document Parsing
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        full_text = ""
        for para in doc.paragraphs:
            full_text += para.text + "\n"
        return full_text[:4000]  # Gemini safe limit
    except Exception as e:
        raise Exception(f"Error extracting text from Word document: {str(e)}")

# Legal Source Finder with Tamil Nadu Support
def get_legal_sources(query, max_results=5):
    # National trusted sites
    national_sites = [
        "site:indiankanoon.org",
        "site:gov.in",
        "site:prsindia.org",
        "site:egazette.nic.in",
        "site:legislative.gov.in",
        "site:india.gov.in",
        "site:legalaffairs.gov.in",
        "site:incometax.gov.in",
        "site:gst.gov.in",
        "site:cbic.gov.in"
    ]
    
    # Tamil Nadu specific sites
    tn_sites = [
        "site:tn.gov.in",
        "site:law.tn.gov.in",
        "site:ctd.tn.gov.in",
        "site:tnreginet.gov.in",
        "site:tnincometax.gov.in",
        "site:tnpolice.gov.in"
    ]
    
    all_sites = national_sites + tn_sites
    combined_sites = " OR ".join(all_sites)

    # Strict domain whitelist (hostnames)
    allowed_domains = [
        "indiankanoon.org",
        "gov.in",
        "prsindia.org",
        "egazette.nic.in",
        "legislative.gov.in",
        "india.gov.in",
        "legalaffairs.gov.in",
        "incometax.gov.in",
        "gst.gov.in",
        "cbic.gov.in",
        "tn.gov.in",
        "law.tn.gov.in",
        "ctd.tn.gov.in",
        "tnreginet.gov.in",
        "tnincometax.gov.in",
        "tnpolice.gov.in",
        "nic.in"
    ]
    search_query = f"{query} {combined_sites}"
    
    try:
        sleep(1)  # Avoid rate limits
        trusted_links = []

        # Primary: DuckDuckGo API wrapper
        try:
            with DDGS() as ddgs:
                results = ddgs.text(search_query, max_results=max_results)
            for result in results:
                href = result.get("href", "")
                if not href:
                    continue
                hostname = urlparse(href).hostname or ""
                if any(hostname.endswith(dom) for dom in allowed_domains):
                    trusted_links.append(href)
        except Exception:
            # Fallback: scrape DuckDuckGo HTML to avoid httpx/proxy incompatibilities
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
                }
                resp = requests.get('https://duckduckgo.com/html/', params={'q': search_query}, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a in soup.select('a.result__a, a[href]'):
                    href = a.get('href', '')
                    if not href or href.startswith('/'):  # skip internal links
                        continue
                    hostname = urlparse(href).hostname or ""
                    if any(hostname.endswith(dom) for dom in allowed_domains):
                        trusted_links.append(href)
                        if len(trusted_links) >= max_results:
                            break
            except Exception:
                pass
        
        # If nothing found, try expanded queries (IPC/section awareness and legal phrasing)
        if not trusted_links:
            expanded_queries = []
            # Section targeting
            section_match = re.search(r'(?:section\s*)?(\d{1,4})', query, re.IGNORECASE)
            if section_match:
                sec = section_match.group(1)
                expanded_queries.extend([
                    f"IPC Section {sec}",
                    f"Indian Penal Code Section {sec}",
                    f"Section {sec} IPC meaning",
                    f"Section {sec} IPC punishment"
                ])
            # IPC expansion
            if re.search(r'\bipc\b', query, re.IGNORECASE):
                expanded_queries.extend([
                    "Indian Penal Code overview",
                    "What is Indian Penal Code"
                ])
            # General legal phrasing
            expanded_queries.extend([
                f"{query} Indian law",
                f"{query} legal definition India"
            ])
            # Try API wrapper first
            try:
                with DDGS() as ddgs:
                    for q in expanded_queries:
                        results2 = ddgs.text(f"{q} {combined_sites}", max_results=max_results)
                        for r in results2:
                            href = r.get("href", "")
                            if not href:
                                continue
                            hostname = urlparse(href).hostname or ""
                            if any(hostname.endswith(dom) for dom in allowed_domains):
                                trusted_links.append(href)
            except Exception:
                # Fallback HTML scraping for expansions
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
                }
                for q in expanded_queries:
                    try:
                        resp = requests.get('https://duckduckgo.com/html/', params={'q': f"{q} {combined_sites}"}, headers=headers, timeout=10)
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        for a in soup.select('a.result__a, a[href]'):
                            href = a.get('href', '')
                            if not href or href.startswith('/'):
                                continue
                            hostname = urlparse(href).hostname or ""
                            if any(hostname.endswith(dom) for dom in allowed_domains):
                                trusted_links.append(href)
                                if len(trusted_links) >= max_results:
                                    break
                    except Exception:
                        continue

        # Deduplicate and cap
        deduped = list(dict.fromkeys(trusted_links))
        return deduped[:max_results]
    except Exception as e:
        raise Exception(f"Error fetching legal sources: {str(e)}")

# Web Scraping
def scrape_web_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text[:4000]
    except Exception:
        return ""

# Summarize Legal Issue
def summarize_legal_issue(document_text, query, urls, model):
    try:
        if not document_text.strip():
            raise Exception("Document is empty.")
        prompt = f"""
You are a legal AI assistant helping citizens understand their rights.

User Question: {query}

Document Content:
{document_text}

Trusted Source URLs:
{', '.join(urls)}

Generate:
- Clear answer with law references
- Steps the citizen can take
- End with source links
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error summarizing document: {str(e)}")


# Simple Legal Query - Direct Gemini Response
def answer_legal_query_directly(query, model):
    try:
        if not query.strip():
            return "Please ask a valid legal question."
        
        # Simple, direct prompt for Gemini
        prompt = f"""You are NeethiAI, a legal AI assistant specializing in Indian law. Answer this question clearly and helpfully:

Question: {query}

Provide a clear, accurate answer about Indian law. If you mention specific laws or sections, be precise. Keep your response conversational and easy to understand."""
        
        response = model.generate_content(prompt)
        return response.text if response.text else "I couldn't generate a response. Please try rephrasing your question."
        
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."

# Tax Advisory
def get_tax_advice(user_query, model):
    try:
        if not user_query.strip():
            raise Exception("Query cannot be empty.")
        system_prompt = (
            "You are a certified Indian tax consultant AI assistant. "
            "Answer the user's question in a clear, structured way using the latest tax laws and budget updates. "
            "Include the following sections:\n\n"
            "1. **User Query**\n"
            "2. **Applicable Tax Regime & Slab**\n"
            "3. **Total Tax Payable (with calculation)**\n"
            "4. **Possible Deductions / Tax Saving Tips**\n"
            "5. **Summary Advice**\n"
            "6. **Trusted Sources** (include official links like incometax.gov.in, cbic.gov.in, or cleartax.in)\n"
            "Use Markdown formatting for headings and bold text."
        )
        full_prompt = f"{system_prompt}\n\nUser: {user_query}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error generating tax advice: {str(e)}")

# Fake Notice Detection
def preprocess_image(image):
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Improve contrast
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_easyocr(image):
    global reader
    try:
        # Use unique temp directory to avoid file locks
        import time
        model_dir = os.path.join(os.getcwd(), 'easyocr_models')
        temp_dir = os.path.join(model_dir, f'temp_{int(time.time() * 1000)}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Detect GPU availability (CUDA) for EasyOCR
        gpu_available = False
        try:
            import torch  # type: ignore
            gpu_available = bool(torch.cuda.is_available())
            if gpu_available:
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                print("No GPU detected via torch.cuda")
        except Exception as e:
            print(f"Torch GPU detection failed: {e}")
            gpu_available = False
        
        # Additional CUDA check
        if not gpu_available:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_available = True
                    print("GPU detected via nvidia-smi")
            except Exception:
                pass
        
        try:
            # For faster processing, use English-only model on CPU
            if not gpu_available:
                print("Using English-only model for faster CPU processing")
                reader = easyocr.Reader(['en'], gpu=False, download_enabled=True, model_storage_directory=temp_dir)
                print("EasyOCR initialized with English-only model (CPU optimized)")
            else:
                # Try multilingual model first (English + Tamil) when GPU is available
                print(f"Initializing EasyOCR with GPU={gpu_available}")
                reader = easyocr.Reader(['en', 'ta'], gpu=gpu_available, download_enabled=True, model_storage_directory=temp_dir)
                print("EasyOCR initialized successfully with multilingual model")
        except Exception as e:
            print(f"Model initialization failed: {e}, falling back to English only")
            # Fallback to English only
            reader = easyocr.Reader(['en'], gpu=False, download_enabled=True, model_storage_directory=temp_dir)
            print("EasyOCR initialized with English-only model (fallback)")
        try:
            results = reader.readtext(image, detail=0, paragraph=True)
        except Exception:
            # Reinitialize with English only and retry
            reader = easyocr.Reader(['en'], gpu=gpu_available, download_enabled=True, model_storage_directory=temp_dir)
            results = reader.readtext(image, detail=0, paragraph=True)
        text = "\n".join([r.strip() for r in results if isinstance(r, str)])
        
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        # Fallback to Tesseract if EasyOCR produced little/no text
        if (not text or len(text.strip()) < 6) and _has_tesseract:
            try:
                if isinstance(image, np.ndarray):
                    pil_img = Image.fromarray(image)
                else:
                    # image could be thresholded numpy already
                    pil_img = Image.fromarray(image)
                t_text = pytesseract.image_to_string(pil_img)
                if t_text and len(t_text.strip()) > len(text.strip() if text else 0):
                    text = t_text
            except Exception:
                pass
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from image: {str(e)}")

def detect_fake_notice(text):
    """Enhanced fake notice detection with multiple fraud patterns"""
    
    # Suspicious keywords and phrases
    suspicious_keywords = [
        "urgent action", "pay immediately", "legal threat", "court summon", 
        "non-bailable", "case filed", "arrest warrant", "pay via wallet",
        "pay via upi", "qr code payment", "immediate payment", "urgent payment",
        "threat", "immediate action", "legal action", "court notice"
    ]
    
    # Suspicious patterns
    suspicious_patterns = [
        r'pay\s+immediately', r'urgent\s+action', r'legal\s+threat',
        r'case\s+filed', r'non-bailable', r'arrest\s+warrant',
        r'pay\s+via\s+(wallet|upi)', r'qr\s+code\s+payment'
    ]
    
    # Government domain patterns (legitimate)
    gov_domains = ['.gov.in', '.nic.in', '.tn.gov.in']
    
    # Check for suspicious keywords
    found_keywords = [kw for kw in suspicious_keywords if kw.lower() in text.lower()]
    
    # Check for suspicious patterns
    found_patterns = []
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            found_patterns.append(pattern)
    
    # Check for missing official elements
    missing_elements = []
    if not re.search(r'fir\s+no|gn\s+no|reference\s+no', text, re.IGNORECASE):
        missing_elements.append("No FIR/GN/Reference number")
    
    if not re.search(r'contact|phone|email|address', text, re.IGNORECASE):
        missing_elements.append("No contact information")
    
    if not re.search(r'government|ministry|department', text, re.IGNORECASE):
        missing_elements.append("No government department mentioned")
    
    # Calculate fraud score
    fraud_score = 0
    fraud_reasons = []
    
    if found_keywords:
        fraud_score += len(found_keywords) * 10
        fraud_reasons.append(f"Suspicious keywords: {', '.join(found_keywords)}")
    
    if found_patterns:
        fraud_score += len(found_patterns) * 15
        fraud_reasons.append(f"Suspicious patterns: {', '.join(found_patterns)}")
    
    if missing_elements:
        fraud_score += len(missing_elements) * 5
        fraud_reasons.append(f"Missing elements: {', '.join(missing_elements)}")
    
    # Check for shortened links (suspicious)
    if re.search(r'bit\.ly|tinyurl|goo\.gl|t\.co', text, re.IGNORECASE):
        fraud_score += 20
        fraud_reasons.append("Contains shortened links (suspicious)")
    
    # Generate result
    if fraud_score >= 30:
        risk_level = "HIGH RISK"
        recommendation = "🚨 **This appears to be a FAKE NOTICE**. Do not make any payments. Report to cybercrime.gov.in"
    elif fraud_score >= 15:
        risk_level = "MEDIUM RISK"
        recommendation = "⚠️ **Exercise caution**. Verify with official sources before taking any action."
    else:
        risk_level = "LOW RISK"
        recommendation = "✅ **Notice appears legitimate**, but always verify with official sources."
    
    result = f"""
## Fraud Detection Analysis

**Risk Level:** {risk_level} (Score: {fraud_score}/100)

**Analysis:**
{chr(10).join(fraud_reasons) if fraud_reasons else "No suspicious patterns detected"}

**Recommendation:**
{recommendation}

**What to do next:**
1. Cross-verify with official government websites
2. Contact the mentioned department directly
3. If suspicious, report to cybercrime.gov.in
4. Never make payments without verification
"""
    
    return result

# Voice Chat Support
def create_voice_chat_interface():
    """Create voice chat interface with HTML/JavaScript"""
    voice_html = """
    <div id="voice-chat-container">
        <button id="start-recording" onclick="startRecording()" style="
            background-color: #003087;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        ">🎤 Start Recording</button>
        
        <button id="stop-recording" onclick="stopRecording()" style="
            background-color: #dc3545;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            display: none;
        ">⏹️ Stop Recording</button>
        
        <button id="play-response" onclick="playResponse()" style="
            background-color: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            display: none;
        ">🔊 Play Response</button>
        
        <div id="transcript" style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; min-height: 50px;">
            <p>Your speech will appear here...</p>
        </div>
        
        <div id="response-text" style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; min-height: 50px; display: none;">
            <p>AI response will appear here...</p>
        </div>
    </div>

    <script>
        let recognition;
        let isRecording = false;
        let responseText = '';

        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
        } else if ('SpeechRecognition' in window) {
            recognition = new SpeechRecognition();
        } else {
            document.getElementById('transcript').innerHTML = '<p>Speech recognition not supported in this browser.</p>';
        }

        if (recognition) {
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US'; // Can be changed to 'ta-IN' for Tamil

            recognition.onstart = function() {
                isRecording = true;
                document.getElementById('start-recording').style.display = 'none';
                document.getElementById('stop-recording').style.display = 'inline-block';
                document.getElementById('transcript').innerHTML = '<p>Listening... Speak now.</p>';
            };

            recognition.onresult = function(event) {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    transcript += event.results[i][0].transcript;
                }
                document.getElementById('transcript').innerHTML = '<p><strong>You said:</strong> ' + transcript + '</p>';
                
                // Send to backend for processing
                if (event.results[i-1].isFinal) {
                    processVoiceInput(transcript);
                }
            };

            recognition.onend = function() {
                isRecording = false;
                document.getElementById('start-recording').style.display = 'inline-block';
                document.getElementById('stop-recording').style.display = 'none';
            };

            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                document.getElementById('transcript').innerHTML = '<p>Error: ' + event.error + '</p>';
            };
        }

        function startRecording() {
            if (recognition && !isRecording) {
                recognition.start();
            }
        }

        function stopRecording() {
            if (recognition && isRecording) {
                recognition.stop();
            }
        }

        function processVoiceInput(transcript) {
            // This would typically send the transcript to your backend
            // For now, we'll just display it
            document.getElementById('response-text').innerHTML = '<p><strong>Processing:</strong> ' + transcript + '</p>';
            document.getElementById('response-text').style.display = 'block';
            
            // In a real implementation, you would send this to your Python backend
            // and get the AI response back
        }

        function playResponse() {
            if ('speechSynthesis' in window && responseText) {
                const utterance = new SpeechSynthesisUtterance(responseText);
                utterance.lang = 'en-US'; // Can be changed to 'ta-IN' for Tamil
                speechSynthesis.speak(utterance);
            }
        }

        // Set response text (called from Python)
        function setResponseText(text) {
            responseText = text;
            document.getElementById('response-text').innerHTML = '<p><strong>AI Response:</strong> ' + text + '</p>';
            document.getElementById('play-response').style.display = 'inline-block';
        }
    </script>
    """
    return voice_html

# Offline Cached Handbook for Common FAQs
def get_offline_handbook():
    """Offline cached handbook for common legal FAQs"""
    return {
        "FIR": {
            "question": "What is an FIR and how to file it?",
            "answer": """
**FIR (First Information Report) - Complete Guide**

**What is FIR?**
- FIR is the first information about a cognizable offence given to a police officer
- It is mandatory for police to register FIR for cognizable offences
- It can be filed by the victim, witness, or any person having knowledge of the crime

**How to file FIR:**
1. Go to the nearest police station
2. Provide detailed information about the incident
3. Police must register FIR within 24 hours
4. Get a copy of the FIR with FIR number
5. If refused, approach Superintendent of Police or file online

**Important Points:**
- FIR is free of cost
- No lawyer required to file FIR
- Police cannot refuse to register FIR for cognizable offences
- You can file FIR online in many states

**Sources:** indiankanoon.org, tnpolice.gov.in
            """,
            "sources": ["https://indiankanoon.org", "https://tnpolice.gov.in"]
        },
        "RTI": {
            "question": "How to file RTI application?",
            "answer": """
**RTI (Right to Information) - Complete Guide**

**What is RTI?**
- RTI Act 2005 gives citizens right to access information from public authorities
- Any citizen can seek information from government departments
- Information must be provided within 30 days (48 hours for life/liberty matters)

**How to file RTI:**
1. Identify the concerned public authority
2. Write application in English/Hindi/local language
3. Pay fee of Rs. 10 (cash/postal order/demand draft)
4. Submit to PIO (Public Information Officer)
5. Get acknowledgment receipt

**Online RTI:**
- Can be filed online at rti.gov.in
- Payment through online banking
- Track status online

**Appeal Process:**
- First appeal to First Appellate Authority within 30 days
- Second appeal to Information Commission within 90 days

**Sources:** rti.gov.in, indiankanoon.org
            """,
            "sources": ["https://rti.gov.in", "https://indiankanoon.org"]
        },
        "GST": {
            "question": "GST registration and filing process",
            "answer": """
**GST (Goods and Services Tax) - Complete Guide**

**GST Registration:**
- Mandatory if annual turnover exceeds Rs. 20 lakhs (Rs. 10 lakhs for special category states)
- Can register voluntarily even below threshold
- Registration is free and online

**GST Filing Process:**
1. **GSTR-1:** Monthly return of outward supplies (by 11th of next month)
2. **GSTR-3B:** Monthly summary return (by 20th of next month)
3. **GSTR-9:** Annual return (by 31st December of next year)

**GST Rates:**
- 0%: Essential items
- 5%: Basic necessities
- 12%: Processed food items
- 18%: Most goods and services
- 28%: Luxury items

**Penalties:**
- Late filing: Rs. 200 per day
- Non-registration: 100% of tax amount

**Sources:** gst.gov.in, cbic.gov.in
            """,
            "sources": ["https://gst.gov.in", "https://cbic.gov.in"]
        },
        "Income Tax": {
            "question": "Income Tax filing and deductions",
            "answer": """
**Income Tax - Complete Guide**

**Tax Slabs (FY 2023-24):**
- Up to Rs. 3 lakh: 0%
- Rs. 3-6 lakh: 5%
- Rs. 6-9 lakh: 10%
- Rs. 9-12 lakh: 15%
- Rs. 12-15 lakh: 20%
- Above Rs. 15 lakh: 30%

**Key Deductions:**
- Section 80C: Up to Rs. 1.5 lakh (EPF, PPF, ELSS, etc.)
- Section 80D: Health insurance premium
- Section 80E: Education loan interest
- Section 24: Home loan interest up to Rs. 2 lakh

**Filing Process:**
1. Calculate total income
2. Claim deductions
3. Calculate tax liability
4. File ITR online before July 31st
5. Pay any balance tax

**Penalties:**
- Late filing: Rs. 1,000-5,000
- Non-filing: Up to Rs. 10,000

**Sources:** incometax.gov.in, cleartax.in
            """,
            "sources": ["https://incometax.gov.in", "https://cleartax.in"]
        }
    }

def search_offline_handbook(query):
    """Search offline handbook for quick answers"""
    handbook = get_offline_handbook()
    query_lower = query.lower()
    
    for key, data in handbook.items():
        if any(word in query_lower for word in [key.lower(), data["question"].lower()]):
            return data
    
    # Search in question content
    for key, data in handbook.items():
        if any(word in query_lower for word in data["question"].lower().split()):
            return data
    
    return None

# Citation Gatekeeper
def citation_gatekeeper(response_text, sources):
    """Ensure all responses have at least one official source"""
    if not sources or len(sources) == 0:
        return "⚠️ **Unable to provide response without verified sources.** Please try rephrasing your question or contact a legal professional directly."
    
    # Check if response contains source links
    has_links = any(domain in response_text.lower() for domain in ['.gov.in', '.nic.in', 'indiankanoon.org', 'prsindia.org'])
    
    if not has_links and sources:
        # Append sources to response
        response_text += "\n\n**📚 Trusted Sources:**\n"
        for i, source in enumerate(sources[:3], 1):
            response_text += f"{i}. [{source}]({source})\n"
    
    return response_text

def enhanced_legal_query_with_gatekeeper(query, model, force_language=None):
    """Enhanced legal query with citation gatekeeper and offline handbook"""
    try:
        if not query.strip():
            raise Exception("Query cannot be empty.")
        
        # First check offline handbook
        offline_result = search_offline_handbook(query)
        if offline_result:
            return citation_gatekeeper(offline_result["answer"], offline_result["sources"])
        
        # If not found in handbook, search online sources
        # If Tamil, translate query to English for better recall
        working_query = query
        if force_language == 'ta' or detect_language(query) == 'ta':
            working_query = translate_text(query, 'en', model)
        urls = get_legal_sources(working_query)
        if not urls:
            return "⚠️ **No verified sources found for your query.** Please try rephrasing or contact a legal professional directly."
        
        for url in urls:
            text = scrape_web_text(url)
            if len(text) > 300:
                # Detect or force language
                language = force_language if force_language in ['ta', 'en'] else detect_language(query)
                lang_prompts = get_language_prompt(language)
                
                prompt = f"""
{lang_prompts['system']}

**User Question:** {query}

**Relevant Legal Content from Trusted Source:**
{text}

**Instructions:**
- {lang_prompts['response_instruction']}
- Provide clear, actionable advice
- Mention relevant legal sections/acts if found
- Keep response concise and citizen-friendly
- Always end with source verification

**Source:** {url}
"""
                response = model.generate_content(
                    contents=[prompt],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2048
                    )
                )
                
                # Apply citation gatekeeper
                final_response = citation_gatekeeper(response.text, [url])
                # If Tamil forced, ensure output in Tamil (translate if needed)
                if force_language == 'ta' and language != 'ta':
                    final_response = translate_text(final_response, 'ta', model)
                return final_response
        
        return "⚠️ **Unable to find reliable legal sources for your query.** Please try rephrasing or contact a legal professional directly."
        
    except Exception as e:
        raise Exception(f"Error answering query: {str(e)}")





# Flask Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('register.html')
        
        # Create new user
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            auth_provider='email'
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Google OAuth Routes
@app.route('/login/google')
def google_login():
    # Check if Google OAuth is properly configured
    if (os.getenv('GOOGLE_CLIENT_ID') == 'your-google-client-id' or 
        os.getenv('GOOGLE_CLIENT_SECRET') == 'your-google-client-secret' or
        not os.getenv('GOOGLE_CLIENT_ID') or 
        not os.getenv('GOOGLE_CLIENT_SECRET')):
        flash('Google OAuth is not configured. Please contact the administrator.')
        return redirect(url_for('login'))
    
    try:
        redirect_uri = url_for('google_callback', _external=True)
        return google.authorize_redirect(redirect_uri)
    except Exception as e:
        flash(f'Google OAuth error: {str(e)}')
        return redirect(url_for('login'))

@app.route('/callback/google')
def google_callback():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        
        if not user_info:
            flash('Failed to get user information from Google')
            return redirect(url_for('login'))
            
        email = user_info.get('email')
        name = user_info.get('name')
        google_id = user_info.get('sub')
        profile_picture = user_info.get('picture')
        
        if not email or not name:
            flash('Incomplete user information from Google')
            return redirect(url_for('login'))
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        if not user:
            # Create new user
            user = User(
                email=email,
                name=name,
                google_id=google_id,
                profile_picture=profile_picture,
                auth_provider='google'
            )
            db.session.add(user)
            db.session.commit()
        else:
            # Update existing user with Google info
            user.google_id = google_id
            user.profile_picture = profile_picture
            user.auth_provider = 'google'
            db.session.commit()
        
        # Login user
        login_user(user)
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        return redirect(url_for('dashboard'))

    except Exception as e:
        print(f"Google OAuth error: {str(e)}")  # Log for debugging
        flash('Google login failed. Please try again or contact support.')
        return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent chats
    recent_chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).limit(10).all()
    return render_template('dashboard.html', chats=recent_chats)

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.get_json()
    question = data.get('question', '').strip()
    force_language = data.get('force_language', '').strip().lower() if isinstance(data.get('force_language', ''), str) else None
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    try:
        # Get Gemini model
        model = configure_gemini_api()
        if not model:
            return jsonify({'error': 'AI service unavailable'}), 500
        
        # Prefer document-grounded answer when user has uploaded a document in this session
        language = force_language if force_language in ['ta', 'en'] else detect_language(question)
        doc_ctx = session.get('doc_context')
        if doc_ctx:
            try:
                prompt = f"""
You are NeethiAI, a legal AI assistant specializing in Indian law.

User question: {question}

Relevant document content:
{doc_ctx[:8000]}

Answer clearly and helpfully. If citing laws/sections, be precise. Keep it user-friendly.
"""
                resp = model.generate_content(prompt)
                answer = resp.text if resp and resp.text else "I couldn't generate a response."
                try:
                    urls = get_legal_sources(question)
                except Exception:
                    urls = []
                sources = urls
                feature_used = 'doc_qa'
            except Exception:
                urls = get_legal_sources(question)
                if urls:
                    answer = enhanced_legal_query_with_gatekeeper(question, model, force_language=force_language)
                    sources = urls
                else:
                    answer = answer_legal_query_directly(question, model)
                    sources = []
                feature_used = 'legal_qa'
        else:
            urls = get_legal_sources(question)
            if urls:
                answer = enhanced_legal_query_with_gatekeeper(question, model, force_language=force_language)
                sources = urls
            else:
                answer = answer_legal_query_directly(question, model)
                sources = []
            feature_used = 'legal_qa'
        
        # Save to database
        chat = Chat(
            user_id=current_user.id,
            question=question,
            answer=answer,
            sources=json.dumps(sources),
            language=language,
            feature_used=feature_used
        )
        db.session.add(chat)
        db.session.commit()
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'chat_id': chat.id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        app.logger.error(f"Chat API error: {str(e)}")
        app.logger.error(f"Error type: {type(e).__name__}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Sorry, I encountered an error. Please try again.'}), 500

@app.route('/api/chat/history')
@login_required
def chat_history():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    chats = Chat.query.filter_by(user_id=current_user.id)\
                     .order_by(Chat.created_at.desc())\
                     .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'chats': [{
            'id': chat.id,
            'question': chat.question,
            'answer': chat.answer,
            'sources': json.loads(chat.sources) if chat.sources else [],
            'language': chat.language,
            'feature_used': chat.feature_used,
            'created_at': chat.created_at.isoformat()
        } for chat in chats.items],
        'has_next': chats.has_next,
        'has_prev': chats.has_prev,
        'page': page,
        'pages': chats.pages
    })

@app.route('/api/chat/<int:chat_id>')
@login_required
def get_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    return jsonify({
        'id': chat.id,
        'question': chat.question,
        'answer': chat.answer,
        'sources': json.loads(chat.sources) if chat.sources else [],
        'language': chat.language,
        'feature_used': chat.feature_used,
        'created_at': chat.created_at.isoformat()
    })

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/api/profile/clear-history', methods=['POST'])
@login_required
def clear_history():
    try:
        Chat.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        # Clear any stored doc context
        session.pop('doc_context', None)
        return jsonify({'ok': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile/change-password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json() or {}
    current = data.get('current_password', '')
    new = data.get('new_password', '')
    if not current or not new:
        return jsonify({'error': 'Both current and new passwords are required'}), 400
    user = User.query.filter_by(id=current_user.id).first()
    if not user or not user.password_hash:
        return jsonify({'error': 'Password change is not available for this account'}), 400
    if not check_password_hash(user.password_hash, current):
        return jsonify({'error': 'Current password is incorrect'}), 400
    user.password_hash = generate_password_hash(new)
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/profile/export', methods=['GET'])
@login_required
def export_chats():
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.asc()).all()
    export = []
    for c in chats:
        export.append({
            'id': c.id,
            'question': c.question,
            'answer': c.answer,
            'sources': json.loads(c.sources) if c.sources else [],
            'language': c.language,
            'feature_used': c.feature_used,
            'created_at': c.created_at.isoformat()
        })
    return jsonify({'chats': export})

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'connected' if os.getenv('DATABASE_URL') else 'not_connected'
    })

# Additional API Routes
@app.route('/api/upload/document', methods=['POST'])
@login_required
def upload_document():
    if 'file' not in request.files:
        app.logger.error('Upload error: no file part in request')
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        app.logger.error('Upload error: empty filename')
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Ensure stream is at start
        try:
            file.stream.seek(0)
        except Exception:
            pass

        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file.stream)
        elif file.filename.lower().endswith('.docx'):
            text = extract_text_from_docx(file.stream)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Initialize AI model for this request
        _model = configure_gemini_api()
        if not _model:
            return jsonify({'error': 'AI service unavailable'}), 500

        # Get legal sources (best effort) and generate summary
        try:
            sources = get_legal_sources("legal document summary")
        except Exception as e:
            app.logger.warning(f'get_legal_sources failed: {e}')
            sources = []
        summary = summarize_legal_issue(text, "Summarize this document", sources, _model)
        
        # Also save a Chat entry for the summary so it appears in history
        summary_chat = Chat(
            user_id=current_user.id,
            question='[Document Uploaded] Summary request',
            answer=summary,
            sources=json.dumps(sources),
            language='en',
            feature_used='doc_summary'
        )
        db.session.add(summary_chat)
        db.session.commit()
        chat_id = summary_chat.id

        return jsonify({
            'text': text,
            'summary': summary,
            'sources': sources,
            'chat_id': chat_id
        })
    except Exception as e:
        app.logger.exception('Upload document failed')
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/fake-notice', methods=['POST'])
@login_required
def detect_fake_notice_api():
    if 'file' not in request.files:
        app.logger.error('Fake-notice upload error: no file part')
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        app.logger.error('Fake-notice upload error: empty filename')
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read raw bytes once for reliability across WSGI servers
        try:
            raw_bytes = file.read()
        except Exception:
            raw_bytes = None
        if not raw_bytes:
            try:
                file.stream.seek(0)
                raw_bytes = file.stream.read()
            except Exception:
                raw_bytes = None
        if not raw_bytes:
            app.logger.error('Fake notice: empty file stream after read')
            return jsonify({'error': 'Could not read uploaded image'}), 400

        # Create a fresh BytesIO for PIL
        img_io = io.BytesIO(raw_bytes)

        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(img_io)
            # Normalize size for better OCR
            if max(image.size) > 1600:
                scale = 1600.0 / max(image.size)
                new_size = (int(image.size[0]*scale), int(image.size[1]*scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            # First try OCR on the original image (RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            np_img = np.array(image)
            text = extract_text_easyocr(np_img)
            if not text or len(text.strip()) < 10:
                # Fallback to preprocessed thresholded image
                processed_image = preprocess_image(image)
                text = extract_text_easyocr(processed_image)
        elif file.filename.lower().endswith('.pdf'):
            # Reuse the raw bytes for PDF parsing
            text = extract_text_from_pdf(io.BytesIO(raw_bytes))
        elif file.filename.lower().endswith('.docx'):
            text = extract_text_from_docx(io.BytesIO(raw_bytes))
        else:
            app.logger.error(f'Unsupported file type: {file.filename}')
            return jsonify({'error': 'Unsupported file type'}), 400
        
        result = detect_fake_notice(text)
        sources = get_legal_sources("fake legal notice")
        
        # Save analysis to chat history
        try:
            saved = Chat(
                user_id=current_user.id,
                question='[Fake Notice] Image checked',
                answer=result,
                sources=json.dumps(sources),
                language='en',
                feature_used='fake_notice'
            )
            db.session.add(saved)
            db.session.commit()
            chat_id = saved.id
        except Exception:
            chat_id = None

        return jsonify({
            'text': text,
            'result': result,
            'sources': sources,
            'chat_id': chat_id
        })
    except Exception as e:
        app.logger.exception('Fake notice detection failed')
        return jsonify({'error': str(e)}), 500

@app.route('/api/tax/advice', methods=['POST'])
@login_required
def tax_advice():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        advice = get_tax_advice(query, model)
        sources = get_legal_sources("tax advisory")
        
        return jsonify({
            'advice': advice,
            'sources': sources
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/handbook/search')
@login_required
def search_handbook():
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    result = search_offline_handbook(query)
    
    if result:
        return jsonify({
            'found': True,
            'question': result['question'],
            'answer': result['answer'],
            'sources': result['sources']
        })
    else:
        return jsonify({
            'found': False,
            'message': 'No matching entry found in the handbook'
        })

# Initialize database and run app
if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 5000))
    
    # Check if running in production
    is_production = os.getenv('FLASK_ENV') == 'production'
    
    # Try to create database tables, but don't fail if database is not available
    try:
        with app.app_context():
            db.create_all()
            print("✅ Database tables created successfully")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        if is_production:
            print("⚠️  Starting in production mode with limited functionality...")
            print("💡 Database will be available when connection is restored")
        else:
            print("⚠️  Continuing in development mode without database...")
    
    if is_production:
        print("🚀 Starting NeethiAI in Production Mode...")
        print(f"🌐 Binding to host: 0.0.0.0, port: {port}")
        # Use waitress for production (stable configuration)
        try:
            from waitress import serve
            print("✅ Waitress imported successfully")
            serve(app, host='0.0.0.0', port=port, threads=4)
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            print("🔄 Falling back to Flask development server...")
            app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
    else:
        print("🚀 Starting NeethiAI Flask Application...")
        print(f"📱 Open your browser and go to: http://localhost:{port}")
        # Run without auto-reloader to avoid upload interruptions
        app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)

