import os
import uuid
import io
import logging

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from google.cloud import speech
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================
# Configuration
# ==========================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define upload folder within the project directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Create directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Flask app settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB limit for video uploads

# Google Cloud Speech-to-Text Credentials
GOOGLE_CREDENTIALS_PATH = os.path.join(BASE_DIR, 'speech-to-text-key.json')
if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    logging.error(f"Google credentials file not found at {GOOGLE_CREDENTIALS_PATH}")
    raise FileNotFoundError(f"Google credentials file not found at {GOOGLE_CREDENTIALS_PATH}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

# ==========================
# Helper Functions
# ==========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, unique_id):
    try:
        video = VideoFileClip(video_path)

        # Check video duration (max 10 minutes = 600 seconds)
        if video.duration > 600:
            logging.error("Video duration exceeds 10 minutes.")
            raise ValueError("Video duration exceeds the maximum allowed limit of 10 minutes.")

        if not hasattr(video, 'fps'):
            logging.error("Video file does not have a valid frame rate.")
            raise ValueError("Invalid video file: no frame rate found.")

        stereo_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
        video.audio.write_audiofile(stereo_audio_path, codec='pcm_s16le')
        logging.info(f"Stereo audio extracted to {stereo_audio_path}")

        mono_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_mono.wav")
        audio = AudioSegment.from_wav(stereo_audio_path)
        audio = audio.set_channels(1)
        audio.export(mono_audio_path, format="wav")
        logging.info(f"Mono audio saved to {mono_audio_path}")

        os.remove(stereo_audio_path)

        video.close()  # Close the video file to free resources
        return mono_audio_path
    except Exception as e:
        logging.exception("Failed to extract and convert audio to mono.")
        raise e

def split_audio(audio_path, chunk_duration=10):
    """Split audio file into smaller chunks of the given duration (in seconds)."""
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration * 1000):  # Convert seconds to milliseconds
        chunk = audio[i:i + chunk_duration * 1000]
        chunk_path = f"{audio_path.rsplit('.', 1)[0]}_chunk_{i // (chunk_duration * 1000)}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio(audio_path):
    try:
        client = speech.SpeechClient()

        # Split audio into chunks
        chunks = split_audio(audio_path)
        transcript = []
        global_start_time = 0  # Track the cumulative start time

        for chunk_path in chunks:
            with io.open(chunk_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code="si-LK",
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,  # Enable word-level timestamps
            )

            response = client.recognize(config=config, audio=audio)

            for result in response.results:
                alternative = result.alternatives[0]
                for word_info in alternative.words:
                    word = word_info.word
                    start_time = global_start_time + word_info.start_time.total_seconds()
                    end_time = global_start_time + word_info.end_time.total_seconds()
                    transcript.append((word, start_time, end_time))

            # Update global start time with the duration of the processed chunk
            chunk_duration = AudioSegment.from_wav(chunk_path).duration_seconds
            global_start_time += chunk_duration

            # Clean up the chunk file
            os.remove(chunk_path)

        logging.info("Transcription completed.")
        return transcript
    except Exception as e:
        logging.exception("Failed to transcribe audio.")
        raise e

def generate_srt_content(transcript):
    """Generate SRT content as a string without saving to file."""
    # Define how many words per subtitle
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
    """Helper function to format time in SRT format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

# ==========================
# Routes
# ==========================

@app.route('/test', methods=['GET'])
def test_srt():
    """Test endpoint that returns a sample SRT content."""
    # Create a sample transcript with word timings
    sample_transcript = [
        ("Hello", 0.0, 1.0),
        ("world", 1.0, 2.0),
        ("this", 2.0, 3.0),
        ("is", 3.0, 3.5),
        ("a", 3.5, 4.0),
        ("test", 4.0, 5.0),
        ("subtitle", 5.0, 6.0),
        ("file", 6.0, 7.0),
        ("with", 7.0, 8.0),
        ("multiple", 8.0, 9.0),
        ("words", 9.0, 10.0),
        ("to", 10.0, 10.5),
        ("check", 10.5, 11.0),
        ("formatting", 11.0, 12.0)
    ]
    
    # Generate SRT content from sample transcript
    srt_content = generate_srt_content(sample_transcript)
    
    return jsonify({
        'srtContent': srt_content,
        'message': 'This is a test SRT content'
    }), 200

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            logging.error('No video part in the request.')
            return jsonify({'error': 'No video part in the request.'}), 400

        file = request.files['video']

        if file.filename == '':
            logging.error('No selected video.')
            return jsonify({'error': 'No selected video.'}), 400

        if file and allowed_file(file.filename):
            # Check file size
            file.seek(0, os.SEEK_END)
            file_length = file.tell()
            if file_length > MAX_VIDEO_SIZE:
                logging.error('File is too large.')
                return jsonify({'error': 'File is too large. Maximum allowed size is 200 MB.'}), 400
            
            file.seek(0)  # Reset file pointer after checking size
            
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            saved_filename = f"{unique_id}_{filename}"
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(saved_path)
            logging.info(f"Video saved to {saved_path}")

            try:
                # Extract audio
                audio_path = extract_audio(saved_path, unique_id)

                # Transcribe audio
                transcript = transcribe_audio(audio_path)

                if not transcript:
                    logging.error("No transcription result.")
                    return jsonify({'error': 'Failed to transcribe audio.'}), 500

                # Generate SRT content directly
                srt_content = generate_srt_content(transcript)

                # Return the SRT content in JSON response
                return jsonify({
                    'srtContent': srt_content
                }), 200

            except Exception as processing_error:
                logging.error(f"Error during video processing: {processing_error}")
                return jsonify({'error': 'Error processing video.'}), 500
            finally:
                # Cleanup: Remove the uploaded video and audio files
                if os.path.exists(saved_path):
                    os.remove(saved_path)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        else:
            logging.error('Unsupported file type.')
            return jsonify({'error': 'Unsupported file type.'}), 400
    except Exception as e:
        logging.exception("Unexpected error during upload.")
        return jsonify({'error': 'Unexpected error occurred.'}), 500

# ==========================
# Main Entry
# ==========================

if __name__ == '__main__':
    app.run(debug=True)
    
