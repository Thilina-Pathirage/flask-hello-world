import io
import json
import logging
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from google.cloud import speech
from flask_cors import CORS
import os
import tempfile
import contextlib

app = Flask(__name__)

# Updated CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["https://unicap.thilina.info"],  # Specify your frontend domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True,
        "max_age": 600  # Cache preflight requests for 10 minutes
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://unicap.thilina.info')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# ==========================
# Configuration
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Allowed video extensions and size limit
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB limit

def setup_google_credentials():
    """Setup Google Cloud credentials from environment variable."""
    try:
        # Get the base64 encoded credentials
        credentials_json_b64 = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        if not credentials_json_b64:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set")

        # Decode the base64 string
        try:
            credentials_json = base64.b64decode(credentials_json_b64).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding in credentials: {str(e)}")

        # Create a temporary file to store the credentials
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write(credentials_json)
            temp_file_path = temp_file.name

        # Set the environment variable to point to the temporary file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
        return temp_file_path

    except Exception as e:
        logging.error(f"Failed to setup Google credentials: {e}")
        raise

# Initialize Google Cloud credentials
try:
    credentials_path = setup_google_credentials()
    # Verify credentials by creating a client
    speech.SpeechClient()
    logging.info("Google Cloud credentials successfully initialized")
except Exception as e:
    logging.error(f"Failed to initialize Google credentials: {e}")
    credentials_path = None

# ==========================
# Helper Functions
# ==========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_stream(video_file_stream):
    """Process video stream using a temporary file."""
    temp_video_path = None
    temp_audio_path = None
    
    try:
        # Create a temporary file and write the video content to it
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            video_file_stream.save(temp_video)
            temp_video_path = temp_video.name

        # Process the video from the temporary file
        video = VideoFileClip(temp_video_path)

        # Check video duration
        if video.duration > 600:
            raise ValueError("Video duration exceeds the maximum allowed limit of 10 minutes.")

        if not hasattr(video, 'fps'):
            raise ValueError("Invalid video file: no frame rate found.")

        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            # Extract audio to temporary file
            video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            
            # Convert to mono using pydub
            audio = AudioSegment.from_wav(temp_audio_path)
            audio = audio.set_channels(1)
            
            mono_stream = io.BytesIO()
            audio.export(mono_stream, format="wav")
            mono_stream.seek(0)

        # Clean up
        video.close()
        return mono_stream

    except Exception as e:
        logging.exception("Failed to process video stream.")
        raise
    finally:
        # Clean up temporary files
        for path in [temp_video_path, temp_audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logging.error(f"Error deleting temporary file {path}: {e}")

def split_audio_stream(audio_stream, chunk_duration=10):
    """Split audio stream into smaller chunks."""
    audio = AudioSegment.from_wav(audio_stream)
    chunks = []
    
    for i in range(0, len(audio), chunk_duration * 1000):
        chunk = audio[i:i + chunk_duration * 1000]
        chunk_stream = io.BytesIO()
        chunk.export(chunk_stream, format="wav")
        chunk_stream.seek(0)
        chunks.append(chunk_stream)
    
    return chunks

def transcribe_audio_stream(audio_stream):
    """Transcribe audio from memory stream."""
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        raise ValueError("Google Cloud credentials not properly configured")

    try:
        client = speech.SpeechClient()
        chunks = split_audio_stream(audio_stream)
        transcript = []
        global_start_time = 0

        for chunk_stream in chunks:
            content = chunk_stream.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code="si-LK",
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True
            )

            response = client.recognize(config=config, audio=audio)

            for result in response.results:
                alternative = result.alternatives[0]
                for word_info in alternative.words:
                    word = word_info.word
                    start_time = global_start_time + word_info.start_time.total_seconds()
                    end_time = global_start_time + word_info.end_time.total_seconds()
                    transcript.append((word, start_time, end_time))

            # Update global start time
            chunk_stream.seek(0)
            chunk_audio = AudioSegment.from_wav(chunk_stream)
            global_start_time += len(chunk_audio) / 1000.0
            chunk_stream.close()

        return transcript

    except Exception as e:
        logging.exception("Failed to transcribe audio.")
        raise

def generate_srt_content(transcript):
    """Generate SRT content as a string."""
    if not transcript:
        return ""
        
    words_per_subtitle = 7
    subtitles = [transcript[i:i + words_per_subtitle] for i in range(0, len(transcript), words_per_subtitle)]

    srt_content = []
    for i, subtitle in enumerate(subtitles):
        start_time = subtitle[0][1]
        end_time = subtitle[-1][2]
        text = ' '.join([word[0] for word in subtitle])

        srt_content.extend([
            str(i + 1),
            f"{format_time(start_time)} --> {format_time(end_time)}",
            f"{text}\n"
        ])

    return '\n'.join(srt_content)

def format_time(seconds):
    """Format time in SRT format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

# ==========================
# Routes
# ==========================

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "message": "Video to SRT converter API is running",
        "credentials_status": "configured" if credentials_path else "not configured"
    })

@app.route('/test-credentials', methods=['GET'])
def test_credentials():
    try:
        if not credentials_path:
            return jsonify({
                'error': 'Credentials not initialized',
                'status': 'failed'
            }), 500

        # Try to create a speech client to verify credentials
        client = speech.SpeechClient()
        
        return jsonify({
            'message': 'Credentials working correctly',
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': f'Error testing credentials: {str(e)}',
            'status': 'failed'
        }), 500

# Handle OPTIONS requests explicitly
@app.route('/upload', methods=['OPTIONS'])
def handle_options():
    return '', 204

@app.route('/upload', methods=['POST'])
def upload_video():
    audio_stream = None
    
    try:
        # Validate credentials
        if not credentials_path:
            return jsonify({'error': 'Google Cloud credentials not configured'}), 500

        # Validate request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected video'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Unsupported file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Check file size
        content_length = request.content_length
        if content_length and content_length > MAX_VIDEO_SIZE:
            return jsonify({
                'error': f'File too large. Maximum size is {MAX_VIDEO_SIZE/1024/1024:.0f}MB'
            }), 400

        # Process video
        audio_stream = process_video_stream(file)
        
        # Transcribe audio
        transcript = transcribe_audio_stream(audio_stream)
        
        if not transcript:
            return jsonify({'error': 'No speech detected in the video'}), 400

        # Generate SRT content
        srt_content = generate_srt_content(transcript)
        
        if not srt_content:
            return jsonify({'error': 'Failed to generate subtitles'}), 500

        return jsonify({
            'srtContent': srt_content,
            'message': 'Video processed successfully'
        }), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.exception("Error processing video")
        return jsonify({'error': str(e)}), 500
    finally:
        if audio_stream:
            try:
                audio_stream.close()
            except Exception:
                pass

if __name__ == '__main__':
    app.run(debug=True)
