import cv2
import numpy as np
import face_recognition
import chromadb
from chromadb.config import Settings
from pathlib import Path
import uuid
from typing import List, Tuple, Optional
import json

class VectorFaceRecognitionSystem:
    """
    Face Recognition System using embeddings stored in ChromaDB vector database
    """
    def __init__(self):
        self.db_dir = Path("face_vector_db")
        self.db_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection for face embeddings
        self.collection = self.client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
        
        print("✓ Vector database initialized")
        self._print_database_stats()
    
    def _print_database_stats(self):
        """Print statistics about the database"""
        count = self.collection.count()
        if count > 0:
            # Get all unique names
            results = self.collection.get()
            names = set([metadata['name'] for metadata in results['metadatas']])
            print(f"✓ Database contains {count} face embeddings for {len(names)} people")
            print(f"  People: {', '.join(sorted(names))}")
        else:
            print("  Database is empty")
    
    def generate_face_embedding(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Generate 128-dimensional face embedding using face_recognition
        
        Args:
            frame: BGR image from OpenCV
            face_location: (top, right, bottom, left) face coordinates
            
        Returns:
            128-dimensional embedding vector or None if failed
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate embedding
        encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        
        if len(encodings) > 0:
            return encodings[0]
        return None
    
    def add_person_to_database(self, person_name: str, num_samples: int = 30):
        """
        Capture face samples and add embeddings to vector database
        
        Args:
            person_name: Name of the person to add
            num_samples: Number of face samples to capture
        """
        print(f"\n=== Adding {person_name} to Database ===")
        print(f"Capturing {num_samples} samples. Look at the camera and move your head slightly.")
        print("Press 'q' to stop early\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        samples_collected = 0
        frame_count = 0
        embeddings_added = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using face_recognition (more accurate than Haar cascades)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            for face_location in face_locations:
                top, right, bottom, left = face_location
                
                # Only capture every 5th frame to get variety
                if frame_count % 5 == 0:
                    # Generate embedding
                    embedding = self.generate_face_embedding(frame, face_location)
                    
                    if embedding is not None:
                        # Add to ChromaDB
                        embedding_id = f"{person_name}_{uuid.uuid4().hex[:8]}"
                        
                        self.collection.add(
                            embeddings=[embedding.tolist()],
                            metadatas=[{
                                "name": person_name,
                                "timestamp": str(frame_count)
                            }],
                            ids=[embedding_id]
                        )
                        
                        samples_collected += 1
                        embeddings_added += 1
                        
                        # Draw green box for captured face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, f"Captured: {samples_collected}/{num_samples}", 
                                  (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Draw yellow box if embedding failed
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                else:
                    # Draw yellow box for detected but not captured
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            
            # Display progress
            cv2.putText(frame, f"Embeddings: {samples_collected}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Adding Person to Database - Press q to stop', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Added {embeddings_added} face embeddings for {person_name} to database")
        self._print_database_stats()
        return embeddings_added > 0
    
    def recognize_face_from_embedding(self, embedding: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Search vector database for similar face embeddings
        
        Args:
            embedding: 128-dimensional face embedding
            threshold: Maximum distance for a match (0-1, lower = stricter)
            
        Returns:
            Tuple of (person_name, similarity_score) or (None, distance) if unknown
        """
        if self.collection.count() == 0:
            return None, 1.0
        
        # Query ChromaDB for similar embeddings
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1
        )
        
        if len(results['distances'][0]) > 0:
            distance = results['distances'][0][0]
            
            # Convert distance to similarity (closer to 1 = more similar)
            similarity = 1 - distance
            
            if distance < threshold:
                person_name = results['metadatas'][0][0]['name']
                return person_name, similarity
        
        return None, 0.0
    
    def recognize_faces_realtime(self, recognition_threshold: float = 0.6):
        """
        Real-time face recognition using vector similarity search
        
        Args:
            recognition_threshold: Maximum distance for recognition (0-1, lower = stricter)
        """
        if self.collection.count() == 0:
            print("\n⚠ Warning: Database is empty! Add some people first.")
            input("Press Enter to continue anyway...")
        
        print("\n=== Face Recognition Started ===")
        print(f"Recognition threshold: {recognition_threshold}")
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 2nd frame for better performance
            if frame_count % 2 == 0:
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect face locations
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    
                    # Generate embedding
                    embedding = self.generate_face_embedding(frame, face_location)
                    
                    if embedding is not None:
                        # Search in vector database
                        person_name, similarity = self.recognize_face_from_embedding(
                            embedding, 
                            threshold=recognition_threshold
                        )
                        
                        if person_name:
                            # Recognized person - green box
                            color = (0, 255, 0)
                            confidence_percent = int(similarity * 100)
                            text = f"{person_name} ({confidence_percent}%)"
                        else:
                            # Unknown person - red box
                            color = (0, 0, 255)
                            text = "Unknown"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw label background
                        cv2.rectangle(frame, (left, top-35), (right, top), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, text, (left+5, top-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition (Vector Search) - Press q to quit', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Face recognition ended")
    
    def list_people(self):
        """List all people in the database"""
        if self.collection.count() == 0:
            print("\n⚠ Database is empty")
            return
        
        results = self.collection.get()
        name_counts = {}
        
        for metadata in results['metadatas']:
            name = metadata['name']
            name_counts[name] = name_counts.get(name, 0) + 1
        
        print("\n=== People in Database ===")
        for name, count in sorted(name_counts.items()):
            print(f"  • {name}: {count} embeddings")
    
    def delete_person(self, person_name: str):
        """Delete all embeddings for a person"""
        results = self.collection.get(
            where={"name": person_name}
        )
        
        if len(results['ids']) == 0:
            print(f"\n⚠ No embeddings found for {person_name}")
            return False
        
        self.collection.delete(ids=results['ids'])
        print(f"\n✓ Deleted {len(results['ids'])} embeddings for {person_name}")
        self._print_database_stats()
        return True
    
    def reset_database(self):
        """Clear all data from the database"""
        confirm = input("\n⚠ This will delete ALL data. Are you sure? (yes/no): ").strip().lower()
        if confirm == 'yes':
            self.client.delete_collection("face_embeddings")
            self.collection = self.client.get_or_create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            print("✓ Database reset complete")
        else:
            print("Reset cancelled")

def main():
    """
    Main menu for vector-based face recognition system
    """
    system = VectorFaceRecognitionSystem()
    
    while True:
        print("\n" + "="*60)
        print("    VECTOR-BASED FACE RECOGNITION SYSTEM")
        print("="*60)
        print("\n1. Add new person to database")
        print("2. Start face recognition (vector search)")
        print("3. List all people in database")
        print("4. Delete person from database")
        print("5. Reset database (clear all data)")
        print("6. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            name = input("\nEnter person's name: ").strip()
            if name:
                num_samples = input("Number of samples (default 30): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 30
                system.add_person_to_database(name, num_samples)
            else:
                print("Invalid name!")
        
        elif choice == '2':
            threshold = input("\nRecognition threshold (0.4-0.7, default 0.6): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.6
                threshold = max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
            except ValueError:
                threshold = 0.6
            system.recognize_faces_realtime(recognition_threshold=threshold)
        
        elif choice == '3':
            system.list_people()
        
        elif choice == '4':
            name = input("\nEnter person's name to delete: ").strip()
            if name:
                system.delete_person(name)
            else:
                print("Invalid name!")
        
        elif choice == '5':
            system.reset_database()
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()
