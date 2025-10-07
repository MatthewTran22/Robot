#!/usr/bin/env python3
"""
Example Usage - Transcription with Speaker Recognition
Shows how to use the transcriber programmatically
"""

from Transcriber import RealtimeTranscriber


def message_handler(text: str):
    """
    Custom callback for handling transcribed messages
    
    Args:
        text: Transcribed text
    """
    # Do something with the transcribed text
    # For example: send to a chatbot, save to database, etc.
    pass


def example_basic_transcription():
    """Example: Basic transcription without speaker recognition"""
    print("=== Basic Transcription (No Speaker Recognition) ===\n")
    
    transcriber = RealtimeTranscriber(
        model_size="base",
        enable_speaker_recognition=False
    )
    
    transcriber.start(language="en", chunk_duration=5)


def example_with_speaker_recognition():
    """Example: Transcription with speaker recognition"""
    print("=== Transcription with Speaker Recognition ===\n")
    
    transcriber = RealtimeTranscriber(
        model_size="base",
        enable_speaker_recognition=True
    )
    
    # Start with callback function
    transcriber.start(
        language="en",
        chunk_duration=5,
        callback=message_handler
    )


def example_custom_settings():
    """Example: Custom transcription settings"""
    print("=== Custom Settings Example ===\n")
    
    transcriber = RealtimeTranscriber(
        model_size="small",          # Better accuracy
        device="cpu",                # or "cuda" for GPU
        compute_type="int8",         # or "float16"/"float32"
        enable_speaker_recognition=True
    )
    
    transcriber.start(
        language="en",               # English
        chunk_duration=3,            # Process every 3 seconds
        callback=message_handler     # Custom handler
    )


if __name__ == "__main__":
    # Choose which example to run
    print("\nExamples:")
    print("1. Basic transcription (no speaker recognition)")
    print("2. Transcription with speaker recognition")
    print("3. Custom settings")
    
    choice = input("\nSelect example (1-3): ").strip()
    
    if choice == '1':
        example_basic_transcription()
    elif choice == '2':
        example_with_speaker_recognition()
    elif choice == '3':
        example_custom_settings()
    else:
        print("Invalid choice")

