"""
FaceRecognition - Face detection and recognition using DeepFace + ChromaDB
Handles face embeddings, vector database, and person identification
"""

import cv2
import numpy as np
from deepface import DeepFace
import chromadb
from chromadb.config import Settings
from pathlib import Path
import uuid
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FaceRecognitionSystem:
    """
    Face Recognition System using DeepFace embeddings stored in ChromaDB vector database
    """
    def __init__(self, model_name: str = "Facenet512"):
        """
        Initialize the face recognition system
        
        Args:
            model_name: DeepFace model to use. Options:
                - "VGG-Face" (2622-d, slow but accurate)
                - "Facenet" (128-d, fast and accurate)
                - "Facenet512" (512-d, best accuracy) - RECOMMENDED
                - "OpenFace" (128-d, lightweight)
                - "DeepFace" (4096-d, very slow)
                - "ArcFace" (512-d, state-of-the-art)
        """
        self.model_name = model_name
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
        
        print(f"✓ Face Recognition initialized with {model_name} model")
        self._print_database_stats()
    
    def _print_database_stats(self):
        """Print statistics about the database"""
        count = self.collection.count()
        if count > 0:
            results = self.collection.get()
            names = set([metadata['name'] for metadata in results['metadatas']])
            print(f"✓ Database contains {count} face embeddings for {len(names)} people")
            print(f"  People: {', '.join(sorted(names))}")
        else:
            print("  Database is empty")
    
    def generate_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using DeepFace
        
        Args:
            face_img: BGR face image from OpenCV
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            embedding_objs = DeepFace.represent(
                img_path=rgb_face,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                return np.array(embedding_objs[0]['embedding'])
            return None
        except Exception:
            return None
    
    def detect_faces(self, frame: np.ndarray, detector: str = "opencv") -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame using DeepFace backends
        
        Args:
            frame: BGR image from OpenCV
            detector: Detection backend ("opencv", "ssd", "mtcnn", "retinaface")
            
        Returns:
            List of (x, y, w, h) tuples for each detected face
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_objs = DeepFace.extract_faces(
                img_path=rgb_frame,
                detector_backend=detector,
                enforce_detection=False,
                align=False
            )
            
            faces = []
            for face_obj in face_objs:
                if face_obj['confidence'] > 0:
                    facial_area = face_obj['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    faces.append((x, y, w, h))
            
            return faces
        except Exception:
            # Fallback to OpenCV Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def recognize_face(self, face_img: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Recognize a face by searching vector database
        
        Args:
            face_img: Face image to recognize
            threshold: Maximum distance for a match (0-1, lower = stricter)
            
        Returns:
            Tuple of (person_name, similarity_score) or (None, 0.0) if unknown
        """
        embedding = self.generate_face_embedding(face_img)
        if embedding is None:
            return None, 0.0
        
        if self.collection.count() == 0:
            return None, 0.0
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1
        )
        
        if len(results['distances'][0]) > 0:
            distance = results['distances'][0][0]
            similarity = 1 - distance
            
            if distance < threshold:
                person_name = results['metadatas'][0][0]['name']
                return person_name, similarity
        
        return None, 0.0
    
    def add_person(self, person_name: str, face_img: np.ndarray) -> bool:
        """
        Add a face embedding to the database
        
        Args:
            person_name: Name of the person
            face_img: Face image
            
        Returns:
            True if added successfully
        """
        embedding = self.generate_face_embedding(face_img)
        if embedding is None:
            return False
        
        embedding_id = f"{person_name}_{uuid.uuid4().hex[:8]}"
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{"name": person_name, "model": self.model_name}],
            ids=[embedding_id]
        )
        return True
    
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
    
    def delete_person(self, person_name: str) -> bool:
        """Delete all embeddings for a person"""
        results = self.collection.get(where={"name": person_name})
        
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

