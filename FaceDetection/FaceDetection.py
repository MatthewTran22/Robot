import cv2
import numpy as np
from deepface import DeepFace
import chromadb
from chromadb.config import Settings
from pathlib import Path
import uuid
from typing import List, Tuple, Optional, Dict
import json
import warnings
import mediapipe as mp
from collections import deque
warnings.filterwarnings('ignore')


class PersonTracker:
    """
    Track a person continuously using OpenCV object tracker, even when face is not visible
    """
    def __init__(self, person_name: str, initial_box: Tuple[int, int, int, int], 
                 frame: np.ndarray, confidence: float, enable_body_tracking: bool = True):
        self.person_name = person_name
        self.confidence = confidence
        self.enable_body_tracking = enable_body_tracking
        self.current_box = initial_box  # Current tracking box
        
        # Initialize OpenCV object tracker (CSRT is accurate, KCF is faster)
        self.tracker = cv2.TrackerKCF_create()  # Fast and reliable
        
        # Expand initial box for full body tracking
        if enable_body_tracking:
            x, y, w, h = initial_box
            # Expand to approximate body size
            body_w = int(w * 2.5)
            body_h = int(h * 5)
            body_x = max(0, x - int((body_w - w) / 2))
            body_y = max(0, y - int(h * 0.5))
            
            # Ensure within frame
            h_frame, w_frame = frame.shape[:2]
            body_w = min(body_w, w_frame - body_x)
            body_h = min(body_h, h_frame - body_y)
            
            self.current_box = (body_x, body_y, body_w, body_h)
        
        # Initialize tracker with the box
        x, y, w, h = self.current_box
        self.tracker.init(frame, (x, y, w, h))
        
        # Face re-identification tracking
        self.frames_since_face_seen = 0
        self.max_frames_without_face = 180  # Re-identify every ~6 seconds (at 30fps)
        self.last_face_box = initial_box
        
    def update_tracking(self, frame: np.ndarray) -> bool:
        """
        Update object tracker position
        
        Returns:
            True if tracking successful, False if lost
        """
        success, box = self.tracker.update(frame)
        
        if success:
            self.current_box = tuple(map(int, box))
            self.frames_since_face_seen += 1
            return True
        else:
            # Tracking lost
            return False
    
    def update_with_face(self, face_box: Tuple[int, int, int, int], 
                         frame: np.ndarray, confidence: float):
        """
        Re-initialize tracker when face is detected again
        """
        self.confidence = confidence
        self.frames_since_face_seen = 0
        self.last_face_box = face_box
        
        # Expand to body box if enabled
        if self.enable_body_tracking:
            x, y, w, h = face_box
            body_w = int(w * 2.5)
            body_h = int(h * 5)
            body_x = max(0, x - int((body_w - w) / 2))
            body_y = max(0, y - int(h * 0.5))
            
            h_frame, w_frame = frame.shape[:2]
            body_w = min(body_w, w_frame - body_x)
            body_h = min(body_h, h_frame - body_y)
            
            new_box = (body_x, body_y, body_w, body_h)
        else:
            new_box = face_box
        
        # Reinitialize tracker with updated position
        self.current_box = new_box
        self.tracker = cv2.TrackerKCF_create()
        x, y, w, h = new_box
        self.tracker.init(frame, (x, y, w, h))
    
    def get_box(self) -> Tuple[int, int, int, int]:
        """Get current tracking box"""
        return self.current_box
    
    def needs_face_reidentification(self) -> bool:
        """Check if we need to re-identify face"""
        return self.frames_since_face_seen > self.max_frames_without_face
    
    def overlaps_with_face(self, face_box: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """
        Check if a detected face overlaps with current tracking box
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            threshold: Minimum IoU overlap threshold
            
        Returns:
            True if face is within tracked region
        """
        x1, y1, w1, h1 = self.current_box
        x2, y2, w2, h2 = face_box
        
        # Calculate intersection over union (IoU)
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return False
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold

class VectorFaceRecognitionSystem:
    """
    Face Recognition System using DeepFace embeddings stored in ChromaDB vector database
    """
    def __init__(self, model_name: str = "Facenet512", enable_body_tracking: bool = True):
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
            enable_body_tracking: Enable full body tracking for recognized faces
        """
        self.model_name = model_name
        self.enable_body_tracking = enable_body_tracking
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
        
        # Initialize MediaPipe Pose for body detection
        if enable_body_tracking:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"✓ Vector database initialized with {model_name} model + Body Tracking")
        else:
            self.pose = None
            print(f"✓ Vector database initialized with {model_name} model")
        
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
    
    def generate_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using DeepFace
        
        Args:
            face_img: BGR face image from OpenCV
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # DeepFace expects RGB
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Generate embedding using DeepFace
            embedding_objs = DeepFace.represent(
                img_path=rgb_face,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='skip'  # We already detected the face
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                return embedding
            return None
        except Exception as e:
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
            # Convert to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using DeepFace
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
        except Exception as e:
            # Fallback to OpenCV Haar Cascade if DeepFace fails
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def get_body_bounding_box(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get full body bounding box using MediaPipe Pose
        
        Args:
            frame: BGR image from OpenCV
            face_box: Face bounding box (x, y, w, h) to help locate the person
            
        Returns:
            Body bounding box (x, y, w, h) or None if detection fails
        """
        if not self.enable_body_tracking or self.pose is None:
            return None
        
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Pose
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Get image dimensions
                h, w, _ = frame.shape
                
                # Extract all landmark coordinates
                landmarks = results.pose_landmarks.landmark
                
                # Get bounding box from all visible landmarks
                x_coords = []
                y_coords = []
                
                for landmark in landmarks:
                    if landmark.visibility > 0.5:  # Only use visible landmarks
                        x_coords.append(int(landmark.x * w))
                        y_coords.append(int(landmark.y * h))
                
                if len(x_coords) > 0 and len(y_coords) > 0:
                    # Calculate bounding box with padding
                    x_min = max(0, min(x_coords) - 20)
                    y_min = max(0, min(y_coords) - 20)
                    x_max = min(w, max(x_coords) + 20)
                    y_max = min(h, max(y_coords) + 20)
                    
                    body_w = x_max - x_min
                    body_h = y_max - y_min
                    
                    # Ensure the body box contains the face box
                    face_x, face_y, face_w, face_h = face_box
                    if x_min <= face_x and y_min <= face_y and x_max >= (face_x + face_w):
                        return (x_min, y_min, body_w, body_h)
            
            # Fallback: estimate body box from face (assume body is ~4x face height)
            face_x, face_y, face_w, face_h = face_box
            h, w, _ = frame.shape
            
            # Estimate body dimensions
            body_w = int(face_w * 2.5)
            body_h = int(face_h * 5)
            
            # Center the body box horizontally around the face
            body_x = max(0, face_x - int((body_w - face_w) / 2))
            body_y = max(0, face_y - int(face_h * 0.5))  # Start slightly above face
            
            # Ensure box stays within frame
            body_w = min(body_w, w - body_x)
            body_h = min(body_h, h - body_y)
            
            return (body_x, body_y, body_w, body_h)
            
        except Exception as e:
            # Fallback estimation
            face_x, face_y, face_w, face_h = face_box
            h, w, _ = frame.shape
            
            body_w = int(face_w * 2.5)
            body_h = int(face_h * 5)
            body_x = max(0, face_x - int((body_w - face_w) / 2))
            body_y = max(0, face_y - int(face_h * 0.5))
            body_w = min(body_w, w - body_x)
            body_h = min(body_h, h - body_y)
            
            return (body_x, body_y, body_w, body_h)
    
    def add_person_to_database(self, person_name: str, num_samples: int = 20, detector: str = "opencv"):
        """
        Capture face samples and add embeddings to vector database
        
        Args:
            person_name: Name of the person to add
            num_samples: Number of face samples to capture
            detector: Face detector to use ("opencv", "ssd", "mtcnn", "retinaface")
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
            
            # Detect faces
            faces = self.detect_faces(frame, detector=detector)
            
            for (x, y, w, h) in faces:
                # Only capture every 5th frame to get variety
                if frame_count % 5 == 0:
                    # Extract face region with some padding
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Generate embedding
                    embedding = self.generate_face_embedding(face_img)
                    
                    if embedding is not None:
                        # Add to ChromaDB
                        embedding_id = f"{person_name}_{uuid.uuid4().hex[:8]}"
                        
                        self.collection.add(
                            embeddings=[embedding.tolist()],
                            metadatas=[{
                                "name": person_name,
                                "timestamp": str(frame_count),
                                "model": self.model_name
                            }],
                            ids=[embedding_id]
                        )
                        
                        samples_collected += 1
                        embeddings_added += 1
                        
                        # Draw green box for captured face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Captured: {samples_collected}/{num_samples}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Draw yellow box if embedding failed
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                else:
                    # Draw yellow box for detected but not captured
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
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
    
    def live_enroll_person(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int], num_samples: int = 10) -> Optional[str]:
        """
        Enroll a new person during live recognition
        
        Args:
            frame: Current frame
            face_coords: Face coordinates (x, y, w, h)
            num_samples: Number of samples to collect
            
        Returns:
            Person name if enrolled successfully, None otherwise
        """
        print("\n=== LIVE ENROLLMENT MODE ===")
        person_name = input("Enter person's name (or press Enter to cancel): ").strip()
        
        if not person_name:
            print("Enrollment cancelled")
            return None
        
        print(f"Enrolling {person_name}... Collecting {num_samples} samples")
        print("Keep your face in frame and move slightly")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access camera")
            return None
        
        samples_collected = 0
        frame_count = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame, detector="opencv")
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Take first face
                
                # Capture every 3rd frame
                if frame_count % 3 == 0:
                    # Extract face with padding
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size > 0:
                        # Generate embedding
                        embedding = self.generate_face_embedding(face_img)
                        
                        if embedding is not None:
                            # Add to database
                            embedding_id = f"{person_name}_{uuid.uuid4().hex[:8]}"
                            self.collection.add(
                                embeddings=[embedding.tolist()],
                                metadatas=[{
                                    "name": person_name,
                                    "timestamp": str(frame_count),
                                    "model": self.model_name
                                }],
                                ids=[embedding_id]
                            )
                            samples_collected += 1
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Display progress
            cv2.putText(frame, f"Enrolling: {person_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{num_samples}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Live Enrollment - Keep face in frame', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if samples_collected >= num_samples:
            print(f"✓ Successfully enrolled {person_name} with {samples_collected} samples!")
            self._print_database_stats()
            return person_name
        else:
            print(f"⚠ Only collected {samples_collected}/{num_samples} samples")
            return None
    
    def recognize_faces_realtime(self, recognition_threshold: float = 0.6, detector: str = "opencv", live_enrollment: bool = True):
        """
        Real-time face recognition using vector similarity search with live enrollment
        
        Args:
            recognition_threshold: Maximum distance for recognition (0-1, lower = stricter)
            detector: Face detector to use ("opencv", "ssd", "mtcnn", "retinaface")
            live_enrollment: Enable pressing 'a' to add new people during recognition
        """
        if self.collection.count() == 0:
            print("\n⚠ Warning: Database is empty!")
            if live_enrollment:
                print("💡 You can press 'a' to add people during recognition")
            input("Press Enter to continue...")
        
        print("\n=== Face Recognition Started ===")
        print(f"Model: {self.model_name}")
        print(f"Detector: {detector}")
        print(f"Body Tracking: {'Enabled' if self.enable_body_tracking else 'Disabled'}")
        print(f"Recognition threshold: {recognition_threshold}")
        print("\n🎮 Controls:")
        print("  'q' - Quit")
        if live_enrollment:
            print("  'a' - Add new person (live enrollment)")
        if self.enable_body_tracking:
            print("\n💡 Recognized people will have FULL BODY bounding boxes!")
        print()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        frame_count = 0
        last_faces = []  # Store last detected faces for enrollment
        tracked_people = []  # List of PersonTracker objects with continuous tracking
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # STEP 1: Update all existing trackers (every frame for smooth continuous tracking)
            trackers_to_remove = []
            for tracker in tracked_people:
                success = tracker.update_tracking(frame)
                if not success:
                    # Tracking failed, mark for removal
                    trackers_to_remove.append(tracker)
            
            # Remove failed trackers
            for tracker in trackers_to_remove:
                tracked_people.remove(tracker)
                print(f"Lost track of {tracker.person_name}")
            
            # STEP 2: Face detection and recognition every N frames
            if frame_count % 10 == 0:  # Check faces every 10 frames
                faces = self.detect_faces(frame, detector=detector)
                last_faces = faces
                
                # Process detected faces
                for (x, y, w, h) in faces:
                    # Extract face region
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Check if this face belongs to an existing tracker
                    matched_tracker = None
                    for tracker in tracked_people:
                        if tracker.overlaps_with_face((x, y, w, h)):
                            matched_tracker = tracker
                            break
                    
                    if matched_tracker:
                        # Update existing tracker with new face detection
                        embedding = self.generate_face_embedding(face_img)
                        if embedding is not None:
                            person_name, similarity = self.recognize_face_from_embedding(
                                embedding, threshold=recognition_threshold
                            )
                            if person_name == matched_tracker.person_name:
                                # Same person, update tracker
                                matched_tracker.update_with_face((x, y, w, h), frame, similarity)
                    else:
                        # New face detected - try to recognize
                        embedding = self.generate_face_embedding(face_img)
                        if embedding is not None:
                            person_name, similarity = self.recognize_face_from_embedding(
                                embedding, threshold=recognition_threshold
                            )
                            
                            if person_name:
                                # Check if we're already tracking this person
                                already_tracked = any(t.person_name == person_name for t in tracked_people)
                                
                                if not already_tracked:
                                    # Create new tracker for recognized person
                                    new_tracker = PersonTracker(
                                        person_name, (x, y, w, h), frame, 
                                        similarity, self.enable_body_tracking
                                    )
                                    tracked_people.append(new_tracker)
                                    print(f"Started tracking {person_name}")
                            else:
                                # Unknown person - just draw box (no tracking)
                                color = (0, 0, 255)
                                text = "Unknown"
                                if live_enrollment:
                                    text = "Unknown (Press 'a' to add)"
                                
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                label_height = 35
                                cv2.rectangle(frame, (x, y-label_height), (x+w, y), color, -1)
                                cv2.putText(frame, text, (x+5, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # STEP 3: Draw all tracked people (every frame - BODY BOX ONLY)
            for tracker in tracked_people:
                x, y, w, h = tracker.get_box()
                
                color = (0, 255, 0)  # Green for recognized and tracked
                text = f"{tracker.person_name}"
                
                # Draw ONLY the body tracking box (thick green line)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
                
                # Draw label at top of body box
                label_height = 40
                cv2.rectangle(frame, (x, y-label_height), (x+w, y), color, -1)
                cv2.putText(frame, text, (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Display instructions
            y_pos = 30
            cv2.putText(frame, "Press 'q' to quit", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if live_enrollment:
                y_pos += 35
                cv2.putText(frame, "Press 'a' to add person", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition (DeepFace + Vector Search)', frame)
            
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a') and live_enrollment:
                # Enter live enrollment mode
                cap.release()
                cv2.destroyAllWindows()
                
                if len(last_faces) > 0:
                    self.live_enroll_person(frame, last_faces[0], num_samples=10)
                else:
                    print("\n⚠ No face detected. Please ensure your face is visible and try again.")
                    input("Press Enter to continue...")
                
                # Restart camera
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not reopen camera")
                    return
        
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
    Main menu for DeepFace vector-based face recognition system
    """
    # Model selection
    print("\n" + "="*60)
    print("    DEEPFACE VECTOR-BASED FACE RECOGNITION SYSTEM")
    print("="*60)
    print("\nAvailable Models:")
    print("1. Facenet512 (512-d, best accuracy) - RECOMMENDED")
    print("2. Facenet (128-d, fast and accurate)")
    print("3. ArcFace (512-d, state-of-the-art)")
    print("4. OpenFace (128-d, lightweight)")
    print("5. VGG-Face (2622-d, slower but accurate)")
    
    model_choice = input("\nSelect model (1-5, default 1): ").strip()
    model_map = {
        '1': 'Facenet512',
        '2': 'Facenet',
        '3': 'ArcFace',
        '4': 'OpenFace',
        '5': 'VGG-Face'
    }
    model_name = model_map.get(model_choice, 'Facenet512')
    
    print(f"\nInitializing with {model_name}...")
    system = VectorFaceRecognitionSystem(model_name=model_name)
    
    while True:
        print("\n" + "="*60)
        print(f"    FACE RECOGNITION SYSTEM ({model_name})")
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
                num_samples = input("Number of samples (default 20): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 20
                
                print("\nDetector options: opencv (fast), ssd, mtcnn, retinaface")
                detector = input("Select detector (default opencv): ").strip() or "opencv"
                
                system.add_person_to_database(name, num_samples, detector=detector)
            else:
                print("Invalid name!")
        
        elif choice == '2':
            threshold = input("\nRecognition threshold (0.4-0.7, default 0.6): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.6
                threshold = max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
            except ValueError:
                threshold = 0.6
            
            print("\nDetector options: opencv (fast), ssd, mtcnn, retinaface")
            detector = input("Select detector (default opencv): ").strip() or "opencv"
            
            system.recognize_faces_realtime(recognition_threshold=threshold, detector=detector)
        
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
