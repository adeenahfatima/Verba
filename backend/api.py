from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import whisper
import nltk
import os
import tempfile
from flask_bcrypt import Bcrypt
from datetime import datetime
import numpy as np
import librosa
from pydub import AudioSegment

# Download NLTK data only if not already present
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Database setup ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///verba.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

bcrypt = Bcrypt(app)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    uploads = db.relationship('Upload', backref='user', lazy=True)

class Upload(db.Model):
    __tablename__ = 'upload'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(256), nullable=False)
    transcript = db.Column(db.Text)
    total_words = db.Column(db.Integer)
    filler_count = db.Column(db.Integer)
    pause_count = db.Column(db.Integer)
    wpm = db.Column(db.Float)
    # Advanced metrics
    pitch_std = db.Column(db.Float)
    pitch_mean = db.Column(db.Float)
    volume_mean = db.Column(db.Float)
    volume_std = db.Column(db.Float)
    noise_level = db.Column(db.Float)
    vocab_richness = db.Column(db.Float)
    advanced_vocab_count = db.Column(db.Integer)
    sentence_var = db.Column(db.Float)
    score = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

# --- End database setup ---

model = whisper.load_model("tiny")  # Load once at startup

def get_float(segment, key):
    value = segment.get(key, 0)
    try:
        return float(value)
    except Exception:
        return 0.0

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Flask backend is running!'})

@app.route('/status', methods=['GET'])
def status():
    try:
        # Check if model is loaded
        if model is not None:
            return jsonify({'model_loaded': True, 'status': 'ready'})
        else:
            return jsonify({'model_loaded': False, 'status': 'loading'})
    except Exception as e:
        return jsonify({'model_loaded': False, 'status': 'error', 'error': str(e)})

@app.route('/transcribe', methods=['GET'])
def transcribe():
    """
    Test route for transcription - this route is deprecated.
    Use /transcribe_upload for proper file processing and database storage.
    """
    return jsonify({
        "error": "This route is deprecated. Use /transcribe_upload for proper file processing.",
        "message": "Upload files through the frontend to get full analysis and database storage."
    }), 400

@app.route('/transcribe_upload', methods=['POST'])
def transcribe_upload():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    if 'audio-upload' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['audio-upload']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Get file extension
        filename = file.filename.lower()
        if filename.endswith(('.mp3', '.m4a', '.aac', '.ogg', '.wma')):
            suffix = '.mp3'
        elif filename.endswith(('.wav')):
            suffix = '.wav'
        else:
            suffix = '.mp3'  # Default to mp3 for unknown formats
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        print(f"Processing audio file: {file.filename} -> {tmp_path}")
        
        # --- Advanced Audio Metrics ---
        pitch_std = pitch_mean = volume_mean = volume_std = noise_level = None
        try:
            import warnings
            warnings.filterwarnings('ignore')
            y, sr = librosa.load(tmp_path, sr=None)
            if y is not None and len(y) > 0:
                # Pitch (F0)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = pitches[magnitudes > np.median(magnitudes)]
                pitch_values = pitch_values[pitch_values > 0]
                pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else None
                pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else None
                # Volume (RMS)
                rms = librosa.feature.rms(y=y)[0]
                volume_mean = float(np.mean(rms)) if len(rms) > 0 else None
                volume_std = float(np.std(rms)) if len(rms) > 0 else None
                # Noise (estimate as low-energy ratio)
                low_energy = np.sum(rms < (0.5 * np.mean(rms))) / len(rms) if len(rms) > 0 else None
                noise_level = float(low_energy) if low_energy is not None else None
        except Exception as audio_err:
            print(f"Audio analysis error: {audio_err}")
            pitch_std = pitch_mean = volume_mean = volume_std = noise_level = None

        # --- Whisper Transcription ---
        result = model.transcribe(tmp_path)
        transcript = result['text']
        if isinstance(transcript, list):
            transcript = " ".join(transcript)
        segments = result['segments']

        filler_words = ["um", "uh", "like", "you know"]
        words = nltk.word_tokenize(transcript.lower())
        total_words = len(words)
        filler_count = sum(words.count(filler) for filler in filler_words)

        pause_count = 0
        for i in range(1, len(segments)):
            gap = get_float(segments[i], 'start') - get_float(segments[i-1], 'end')
            if gap > 2:
                pause_count += 1

        duration = get_float(segments[-1], 'end') if segments else 1.0
        wpm = (total_words / duration) * 60 if duration > 0 else 0

        # --- Advanced Transcript Metrics ---
        vocab_richness = advanced_vocab_count = sentence_var = None
        try:
            # Vocabulary richness
            unique_words = set(words)
            vocab_richness = len(unique_words) / total_words if total_words > 0 else None
            # Advanced vocabulary (not in top 2000 common English words)
            common_words = set(nltk.corpus.words.words()[:2000])
            advanced_words = [w for w in unique_words if w.isalpha() and w not in common_words]
            advanced_vocab_count = len(advanced_words)
            # Sentence structure
            sentences = nltk.sent_tokenize(transcript)
            sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
            sentence_var = float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else None
        except Exception as text_err:
            print(f"Transcript analysis error: {text_err}")
            vocab_richness = advanced_vocab_count = sentence_var = None

        # --- Composite Score (out of 100) ---
        score = 0
        try:
            # WPM: ideal 110-160
            if 110 <= wpm <= 160:
                score += 15
            elif 90 <= wpm < 110 or 160 < wpm <= 180:
                score += 10
            elif 70 <= wpm < 90 or 180 < wpm <= 200:
                score += 5
            # Filler words: fewer is better
            if filler_count == 0:
                score += 15
            elif filler_count <= 2:
                score += 10
            elif filler_count <= 5:
                score += 5
            # Pauses: 0-2 is best
            if pause_count <= 2:
                score += 10
            elif pause_count <= 5:
                score += 5
            # Pitch variation: higher is better
            if pitch_std is not None:
                if pitch_std > 20:
                    score += 10
                elif pitch_std > 10:
                    score += 7
                elif pitch_std > 5:
                    score += 4
            # Volume consistency: lower std is better
            if volume_std is not None:
                if volume_std < 0.01:
                    score += 10
                elif volume_std < 0.03:
                    score += 7
                elif volume_std < 0.05:
                    score += 4
            # Noise: lower is better
            if noise_level is not None:
                if noise_level < 0.2:
                    score += 10
                elif noise_level < 0.4:
                    score += 7
                elif noise_level < 0.6:
                    score += 4
            # Vocabulary richness
            if vocab_richness is not None:
                if vocab_richness > 0.5:
                    score += 10
                elif vocab_richness > 0.3:
                    score += 7
                elif vocab_richness > 0.15:
                    score += 4
            # Advanced vocab
            if advanced_vocab_count is not None:
                if advanced_vocab_count > 10:
                    score += 5
                elif advanced_vocab_count > 5:
                    score += 3
            # Sentence structure
            if sentence_var is not None:
                if sentence_var > 5:
                    score += 5
                elif sentence_var > 2:
                    score += 3
        except Exception as score_err:
            print(f"Score calculation error: {score_err}")
            score = 0

        print(f"Analysis complete: {total_words} words, {filler_count} fillers, {pause_count} pauses, {wpm:.2f} wpm, pitch std: {pitch_std}, volume std: {volume_std}, noise: {noise_level}, vocab richness: {vocab_richness}, advanced vocab: {advanced_vocab_count}, sentence var: {sentence_var}, score: {score}")

        # Save to database (now with advanced metrics)
        upload = Upload(
            user_id=user_id,
            filename=file.filename,
            transcript=transcript,
            total_words=total_words,
            filler_count=filler_count,
            pause_count=pause_count,
            wpm=round(wpm, 2),
            pitch_std=pitch_std,
            pitch_mean=pitch_mean,
            volume_mean=volume_mean,
            volume_std=volume_std,
            noise_level=noise_level,
            vocab_richness=vocab_richness,
            advanced_vocab_count=advanced_vocab_count,
            sentence_var=sentence_var,
            score=score
        )
        db.session.add(upload)
        db.session.commit()

        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass  # Ignore cleanup errors

        return jsonify({
            "transcript": transcript,
            "total_words": total_words,
            "filler_count": filler_count,
            "pause_count": pause_count,
            "wpm": round(wpm, 2),
            "pitch_std": pitch_std,
            "pitch_mean": pitch_mean,
            "volume_mean": volume_mean,
            "volume_std": volume_std,
            "noise_level": noise_level,
            "vocab_richness": vocab_richness,
            "advanced_vocab_count": advanced_vocab_count,
            "sentence_var": sentence_var,
            "score": score
        })
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return jsonify({"error": "Audio processing failed. Please try a different or clearer audio file."}), 500

@app.route('/uploads/<int:user_id>', methods=['GET'])
def get_uploads(user_id):
    uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.timestamp.desc()).all()
    return jsonify([
        {
            'id': u.id,
            'filename': u.filename,
            'transcript': u.transcript,
            'total_words': u.total_words,
            'filler_count': u.filler_count,
            'pause_count': u.pause_count,
            'wpm': u.wpm,
            'pitch_std': u.pitch_std,
            'pitch_mean': u.pitch_mean,
            'volume_mean': u.volume_mean,
            'volume_std': u.volume_std,
            'noise_level': u.noise_level,
            'vocab_richness': u.vocab_richness,
            'advanced_vocab_count': u.advanced_vocab_count,
            'sentence_var': u.sentence_var,
            'score': u.score,
            'timestamp': u.timestamp.isoformat() if u.timestamp else None
        } for u in uploads
    ])

@app.route('/profile/<int:user_id>', methods=['GET'])
def get_profile(user_id):
    uploads = Upload.query.filter_by(user_id=user_id).all()
    total_uploads = len(uploads)
    best_wpm = max((u.wpm for u in uploads if u.wpm is not None), default=0)
    lowest_filler = min((u.filler_count for u in uploads if u.filler_count is not None), default=0)
    longest_speech = max((u.total_words for u in uploads if u.total_words is not None), default=0)
    return jsonify({
        'total_uploads': total_uploads,
        'best_wpm': best_wpm,
        'lowest_filler': lowest_filler,
        'longest_speech': longest_speech
    })

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not email or not password:
        return jsonify({'error': 'Missing required fields'}), 400
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'error': 'Username or email already exists'}), 400
    pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, email=email, password_hash=pw_hash)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username_or_email = data.get('username') or data.get('email')
    password = data.get('password')
    if not username_or_email or not password:
        return jsonify({'error': 'Missing required fields'}), 400
    user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid username/email or password'}), 401
    return jsonify({'message': 'Login successful', 'user': {'id': user.id, 'username': user.username, 'email': user.email}})

if __name__ == '__main__':
    print("Flask app loaded")
    print("Starting Flask server...")
    print("Model loading...")
    print("Server ready!")
    print(f"Current working directory: {os.getcwd()}")
    try:
        with app.app_context():
            db.create_all()
        print("Database tables created (or already exist).")
    except Exception as e:
        print(f"Error during db.create_all(): {e}")
    app.run(debug=True, host='127.0.0.1', port=5000)
