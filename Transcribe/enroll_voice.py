#!/usr/bin/env python3
"""
Voice Enrollment Tool
Record and save voice samples for speaker recognition
"""

import pyaudio
import wave
import time
from VoiceRecognition import VoiceRecognizer


def record_audio(filename: str, duration: int = 5) -> str:
    """
    Record audio from microphone
    
    Args:
        filename: Output filename
        duration: Recording duration in seconds
    
    Returns:
        Path to saved audio file
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    audio = pyaudio.PyAudio()
    
    print(f"\n🎤 Recording for {duration} seconds...")
    print("Please speak clearly into your microphone.")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("🔴 RECORDING NOW!\n")
    
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    print("✓ Recording complete!\n")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save to file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename


def main():
    """Main enrollment application"""
    print("\n" + "="*60)
    print("    VOICE ENROLLMENT - SPEAKER REGISTRATION")
    print("="*60)
    
    recognizer = VoiceRecognizer()
    
    # Show currently enrolled speakers
    recognizer.list_enrolled_speakers()
    
    print("\n" + "-"*60)
    print("This tool will record your voice and save it for recognition")
    print("-"*60)
    
    while True:
        print("\nOptions:")
        print("1. Enroll new speaker (record voice)")
        print("2. Enroll from existing audio file")
        print("3. Remove speaker")
        print("4. List enrolled speakers")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            name = input("\nEnter speaker name: ").strip()
            if not name:
                print("Name cannot be empty!")
                continue
            
            duration = input("Recording duration in seconds (default 5): ").strip()
            duration = int(duration) if duration.isdigit() else 5
            
            # Record audio
            temp_file = f"temp_{name}_enrollment.wav"
            record_audio(temp_file, duration)
            
            # Enroll speaker
            success = recognizer.enroll_speaker(name, temp_file)
            
            if success:
                print(f"\n✅ {name} has been enrolled successfully!")
            
            # Clean up temp file
            import os
            try:
                os.remove(temp_file)
            except:
                pass
        
        elif choice == '2':
            name = input("\nEnter speaker name: ").strip()
            audio_path = input("Enter path to audio file (WAV): ").strip()
            
            recognizer.enroll_speaker(name, audio_path)
        
        elif choice == '3':
            name = input("\nEnter speaker name to remove: ").strip()
            recognizer.remove_speaker(name)
        
        elif choice == '4':
            recognizer.list_enrolled_speakers()
        
        elif choice == '5':
            print("\n✓ Enrollment complete. Goodbye!")
            break
        
        else:
            print("Invalid option, please try again")


if __name__ == "__main__":
    main()

