"""
VoiceRecognition - Speaker Identification using Resemblyzer
Identifies speakers by their voice characteristics
"""

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import pickle
import os
from typing import Optional, Dict, Tuple
from pathlib import Path


class VoiceRecognizer:
    """
    Speaker identification using voice embeddings
    """
    def __init__(self, database_path: str = "voice_database.pkl", threshold: float = 0.75):
        """
        Initialize the voice recognizer
        
        Args:
            database_path: Path to save/load voice profiles
            threshold: Similarity threshold for recognition (0-1, higher = stricter)
        """
        self.encoder = VoiceEncoder()
        self.database_path = database_path
        self.threshold = threshold
        self.voice_profiles: Dict[str, np.ndarray] = {}
        
        # Load existing database if available
        self.load_database()
    
    def enroll_speaker(self, name: str, audio_path: str) -> bool:
        """
        Enroll a new speaker by creating their voice profile
        
        Args:
            name: Speaker's name
            audio_path: Path to audio file (WAV) with speaker's voice
            
        Returns:
            True if enrollment successful
        """
        try:
            # Load and preprocess audio
            wav = preprocess_wav(audio_path)
            
            # Create voice embedding
            embedding = self.encoder.embed_utterance(wav)
            
            # Save to profiles
            self.voice_profiles[name] = embedding
            self.save_database()
            
            print(f"✓ Enrolled '{name}' successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error enrolling '{name}': {e}")
            return False
    
    def identify_speaker(self, audio_path: str) -> Tuple[Optional[str], float]:
        """
        Identify a speaker from audio
        
        Args:
            audio_path: Path to audio file to identify
            
        Returns:
            Tuple of (speaker_name, confidence) or (None, 0) if unknown
        """
        if not self.voice_profiles:
            return None, 0.0
        
        try:
            # Load and preprocess audio
            wav = preprocess_wav(audio_path)
            
            # Create voice embedding
            embedding = self.encoder.embed_utterance(wav)
            
            # Compare with all profiles
            best_match = None
            best_similarity = 0.0
            
            for name, profile_embedding in self.voice_profiles.items():
                # Calculate cosine similarity
                similarity = np.dot(embedding, profile_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(profile_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Check if similarity meets threshold
            if best_similarity >= self.threshold:
                return best_match, float(best_similarity)
            else:
                return "Unknown", float(best_similarity)
                
        except Exception as e:
            print(f"Error identifying speaker: {e}")
            return None, 0.0
    
    def save_database(self):
        """Save voice profiles to disk"""
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.voice_profiles, f)
        print(f"✓ Voice database saved to {self.database_path}")
    
    def load_database(self):
        """Load voice profiles from disk"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    self.voice_profiles = pickle.load(f)
                print(f"✓ Loaded {len(self.voice_profiles)} voice profile(s)")
            except Exception as e:
                print(f"Warning: Could not load database: {e}")
                self.voice_profiles = {}
        else:
            print("No existing voice database found")
    
    def list_enrolled_speakers(self):
        """Print list of enrolled speakers"""
        if self.voice_profiles:
            print("\nEnrolled Speakers:")
            for i, name in enumerate(self.voice_profiles.keys(), 1):
                print(f"  {i}. {name}")
        else:
            print("No speakers enrolled yet")
    
    def remove_speaker(self, name: str) -> bool:
        """Remove a speaker from the database"""
        if name in self.voice_profiles:
            del self.voice_profiles[name]
            self.save_database()
            print(f"✓ Removed '{name}'")
            return True
        else:
            print(f"✗ '{name}' not found in database")
            return False


def main():
    """Voice enrollment utility"""
    print("\n" + "="*60)
    print("    VOICE RECOGNITION - SPEAKER ENROLLMENT")
    print("="*60)
    
    recognizer = VoiceRecognizer()
    recognizer.list_enrolled_speakers()
    
    print("\nOptions:")
    print("1. Enroll new speaker")
    print("2. Test speaker identification")
    print("3. Remove speaker")
    print("4. List enrolled speakers")
    print("5. Exit")
    
    while True:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            name = input("Enter speaker name: ").strip()
            audio_path = input("Enter path to audio file (WAV): ").strip()
            recognizer.enroll_speaker(name, audio_path)
        
        elif choice == '2':
            audio_path = input("Enter path to audio file to identify: ").strip()
            speaker, confidence = recognizer.identify_speaker(audio_path)
            if speaker:
                print(f"\n🎯 Identified: {speaker} (confidence: {confidence:.2%})")
            else:
                print("\n✗ Could not identify speaker")
        
        elif choice == '3':
            name = input("Enter speaker name to remove: ").strip()
            recognizer.remove_speaker(name)
        
        elif choice == '4':
            recognizer.list_enrolled_speakers()
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()

