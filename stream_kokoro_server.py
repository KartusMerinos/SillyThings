from flask import Flask, request, jsonify, Response
import subprocess
import os
import threading
import queue
import json
import sounddevice as sd
import numpy as np
import requests
import time
import tempfile
import wave
from threading import Lock

app = Flask(__name__)

# Configuration
ALLOWED_MODELS = ["tts-1", "tts-1-hd"]
VOICE_OPTIONS = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia", "af_jessica", "af_kore", "af_nicole", 
    "af_nova", "af_river", "af_sarah", "af_sky", "af_v0bella", "af_v0irulan", "af_v0nicole", "af_v0", 
    "af_v0sarah", "af_v0sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", 
    "am_onyx", "am_puck", "am_santa", "am_v0adam", "am_v0gurney", "am_v0michael", "bf_alice", "bf_emma", 
    "bf_lily", "bf_v0emma", "bf_v0isabella", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "bm_v0george", 
    "bm_v0lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", 
    "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", 
    "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxia", 
    "zm_yunxi", "zm_yunyang"
]

# Global playback speed setting
PLAYBACK_SPEED = 1.7

# Global audio queue and playback thread state
audio_queue = queue.Queue()
is_playing = False
playback_lock = Lock()

def audio_player_thread():
    """Thread to continuously play audio chunks from the queue"""
    global is_playing
    
    print("Audio player thread started")
    while True:
        try:
            # Get the next audio chunk from the queue
            audio_data, sample_rate = audio_queue.get()
            
            with playback_lock:
                is_playing = True
            
            # Play the audio
            print(f"Playing chunk: {len(audio_data)} samples at {sample_rate}Hz")
            sd.play(audio_data, sample_rate, blocking=True)
            
            # Mark task as done
            audio_queue.task_done()
            
        except Exception as e:
            print(f"Error in audio player thread: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            with playback_lock:
                is_playing = False

# Start the audio player thread
player_thread = threading.Thread(target=audio_player_thread, daemon=True)
player_thread.start()

def process_audio_chunk(audio_content, speed=PLAYBACK_SPEED):
    """Process a single audio chunk with FFmpeg and add to playback queue"""
    try:
        # Create temp files with unique names
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_input_file:
            temp_input = temp_input_file.name
            temp_input_file.write(audio_content)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output_file:
            temp_output = temp_output_file.name
        
        # Use FFmpeg to speed up audio without changing pitch
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_input, 
            "-filter:a", f"atempo={speed}", 
            "-ar", "24000",  # Ensure consistent sample rate
            temp_output
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode('utf-8')}")
            raise Exception("FFmpeg processing failed")
            
        # Load the processed audio
        with wave.open(temp_output, 'rb') as wf:
            # Get audio parameters
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
            
            # Read all frames
            audio_data = wf.readframes(frames)
            
            # Convert to numpy array for playback
            if width == 2:  # 16-bit audio
                dtype = np.int16
            elif width == 4:  # 32-bit audio
                dtype = np.int32
            else:
                dtype = np.uint8
                
            audio_array = np.frombuffer(audio_data, dtype=dtype)
            
            # Add to playback queue
            audio_queue.put((audio_array, rate))
            
        # Clean up temporary files
        try:
            os.remove(temp_input)
            os.remove(temp_output)
        except Exception as e:
            print(f"Error cleaning up temp files: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"Error processing audio chunk: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def chunk_text(text, max_chunk_size=500):
    """Split text into reasonable chunks at sentence boundaries"""
    # Simple sentence splitting - can be improved
    sentences = []
    current = ""
    for char in text:
        current += char
        # Split on sentence endings
        if char in ['.', '!', '?'] and len(current.strip()) > 0:
            sentences.append(current)
            current = ""
    
    # Add any remaining text
    if current:
        sentences.append(current)
    
    # Combine sentences into chunks of appropriate size
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def process_text_chunks(text, voice, speed=1.0):
    """Process text in chunks and stream each chunk as it's ready"""
    chunks = chunk_text(text)
    print(f"Text split into {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:30]}...")
            
            # Request TTS for this chunk
            response = requests.post(
                "http://localhost:8880/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": chunk,
                    "voice": voice,
                    "speed": speed,
                    "response_format": "mp3",
                },
                timeout=300  # Shorter timeout for chunks
            )
            response.raise_for_status()
            
            # Process and queue the audio
            success = process_audio_chunk(response.content)
            if not success:
                print(f"Failed to process chunk {i+1}")
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()

@app.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    try:
        data = request.get_json()

        # Validate required fields
        if not all(key in data for key in ['model', 'input', 'voice']):
            return jsonify({
                'error': {
                    'message': 'Missing required fields. Required: model, input, voice',
                    'type': 'invalid_request_error'
                }
            }), 400

        # Validate voice
        if data['voice'] not in VOICE_OPTIONS:
            return jsonify({
                'error': {
                    'message': f'Invalid voice. Allowed voices: {", ".join(VOICE_OPTIONS)}',
                    'type': 'invalid_request_error'
                }
            }), 400

        text = data['input']
        voice = data['voice']
        speed = data.get('speed', 1.0)

        # Clear existing queue if there's anything left
        with playback_lock:
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                except queue.Empty:
                    break

        # Process chunks in a separate thread
        processing_thread = threading.Thread(
            target=process_text_chunks,
            args=(text, voice, speed)
        )
        processing_thread.daemon = True
        processing_thread.start()

        # Return success response immediately
        return jsonify({
            'success': True,
            'message': f'Audio streaming started at {PLAYBACK_SPEED}x playback speed with chunked processing',
            'details': {
                'text_length': len(text),
                'voice': voice,
                'tts_speed': speed,
                'playback_speed': PLAYBACK_SPEED,
                'method': 'Streaming chunks with FFmpeg processing'
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': {
                'message': f'Error processing request: {str(e)}',
                'type': 'server_error'
            }
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFmpeg detected - using optimal audio processing")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("WARNING: FFmpeg not found - audio processing may be limited")
    
    app.run(host='0.0.0.0', port=8002, debug=True) 
This is my modification with a playback speed of 1.7x using Claude. Thank you for your guidance—much appreciated.
