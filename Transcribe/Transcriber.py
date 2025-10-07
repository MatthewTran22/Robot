"""
Transcriber - Real-time Speech-to-Text using Faster-Whisper
Continuous audio transcription for live applications with speaker recognition
"""

import pyaudio
import wave
from faster_whisper import WhisperModel
import threading
import queue
from typing import Optional, Callable
import time
from VoiceRecognition import VoiceRecognizer


class RealtimeTranscriber:
    """
    Real-time audio transcription using Faster-Whisper
    """
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8",
                 enable_speaker_recognition: bool = True):
        """
        Initialize the transcriber
        
        Args:
            model_size: Model size - "tiny", "base", "small", "medium", "large-v3"
            device: "cpu" or "cuda" for GPU acceleration
            compute_type: "int8" (fast, CPU), "float16" (GPU), "float32" (best quality)
            enable_speaker_recognition: Enable speaker identification
        """
        print(f"Loading Faster-Whisper model '{model_size}'...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("✓ Model loaded successfully")
        
        # Audio configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Voice recognition
        self.enable_speaker_recognition = enable_speaker_recognition
        self.voice_recognizer = None
        if enable_speaker_recognition:
            try:
                self.voice_recognizer = VoiceRecognizer()
                if self.voice_recognizer.voice_profiles:
                    print(f"✓ Voice recognition enabled ({len(self.voice_recognizer.voice_profiles)} speaker(s) enrolled)")
                else:
                    print("⚠ Voice recognition enabled but no speakers enrolled yet")
                    print("  Run 'python Transcribe/enroll_voice.py' to enroll speakers")
            except Exception as e:
                print(f"⚠ Could not load voice recognition: {e}")
                self.enable_speaker_recognition = False
        
    def start(self, callback: Optional[Callable[[str], None]] = None,
              language: Optional[str] = None, 
              chunk_duration: int = 5):
        """
        Start real-time transcription from microphone
        
        Args:
            callback: Function to call with transcribed text
            language: Language code (e.g., "en", "es", "fr") or None for auto-detect
            chunk_duration: Process audio in chunks of N seconds
        """
        if self.is_recording:
            print("Already recording!")
            return
        
        self.is_recording = True
        print("\n🎤 Real-time transcription started")
        print("Press Ctrl+C to stop\n")
        
        # Start recording thread
        record_thread = threading.Thread(
            target=self._record_loop,
            args=(chunk_duration,)
        )
        record_thread.daemon = True
        record_thread.start()
        
        # Process audio chunks
        try:
            while self.is_recording:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # Save chunk temporarily
                    temp_file = "temp_chunk.wav"
                    wf = wave.open(temp_file, 'wb')
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(audio_data)
                    wf.close()
                    
                    # Transcribe chunk
                    segments, _ = self.model.transcribe(
                        temp_file,
                        language=language,
                        beam_size=5,
                        vad_filter=True,  # Voice Activity Detection
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # Identify speaker if enabled
                    speaker_name = "Unknown"
                    confidence = 0.0
                    if self.enable_speaker_recognition and self.voice_recognizer:
                        speaker_name, confidence = self.voice_recognizer.identify_speaker(temp_file)
                    
                    for segment in segments:
                        text = segment.text.strip()
                        if text:
                            timestamp = time.strftime("%H:%M:%S")
                            
                            # Format output as "name: message"
                            if self.enable_speaker_recognition:
                                confidence_indicator = f" ({confidence:.0%})" if confidence > 0 else ""
                                print(f"[{timestamp}] {speaker_name}{confidence_indicator}: {text}")
                            else:
                                print(f"[{timestamp}] {text}")
                            
                            if callback:
                                callback(text)
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n✓ Transcription stopped")
            self.stop()
    
    def _record_loop(self, chunk_duration: int):
        """Internal recording loop"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        while self.is_recording:
            frames = []
            for _ in range(0, int(self.RATE / self.CHUNK * chunk_duration)):
                if not self.is_recording:
                    break
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            if frames:
                self.audio_queue.put(b''.join(frames))
        
        stream.stop_stream()
        stream.close()
    
    def stop(self):
        """Stop real-time transcription"""
        self.is_recording = False
    
    def __del__(self):
        """Cleanup"""
        self.audio.terminate()


def main():
    """Real-time transcription application"""
    print("\n" + "="*60)
    print("    REAL-TIME SPEECH-TO-TEXT TRANSCRIBER")
    print("    With Speaker Recognition")
    print("="*60)
    print("\nAvailable Models:")
    print("1. tiny   - Fastest, basic accuracy")
    print("2. base   - Good balance (RECOMMENDED)")
    print("3. small  - Better accuracy, slower")
    print("4. medium - High accuracy, much slower")
    print("5. large  - Best accuracy, very slow")
    
    model_choice = input("\nSelect model (1-5, default 2): ").strip()
    model_map = {
        '1': 'tiny',
        '2': 'base',
        '3': 'small',
        '4': 'medium',
        '5': 'large-v3'
    }
    model_size = model_map.get(model_choice, 'base')
    
    print("\nLanguage (press Enter for auto-detect, or enter code like 'en', 'es', 'fr')")
    language = input("Language: ").strip() or None
    
    chunk_duration = input("Process chunks of N seconds (default 5): ").strip()
    chunk_duration = int(chunk_duration) if chunk_duration.isdigit() else 5
    
    print("\nEnable speaker recognition? (y/n, default y): ", end='')
    enable_recognition = input().strip().lower() != 'n'
    
    print(f"\nInitializing with '{model_size}' model...")
    transcriber = RealtimeTranscriber(
        model_size=model_size,
        enable_speaker_recognition=enable_recognition
    )
    
    # Start transcription
    transcriber.start(
        language=language,
        chunk_duration=chunk_duration
    )


if __name__ == "__main__":
    main()

