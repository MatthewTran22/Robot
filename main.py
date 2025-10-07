"""
Main Application - Face Detection with Continuous Person Tracking
Orchestrates face recognition and object tracking
"""

import cv2
from FaceDetection.PersonTracker import PersonTracker
from FaceDetection.FaceRecognition import FaceRecognitionSystem


class FaceDetectionApp:
    """
    Main application combining face recognition with continuous person tracking
    """
    def __init__(self, model_name: str = "Facenet512", enable_body_tracking: bool = True):
        """
        Initialize the application
        
        Args:
            model_name: DeepFace model to use
            enable_body_tracking: Enable full body tracking boxes
        """
        self.face_recognition = FaceRecognitionSystem(model_name=model_name)
        self.enable_body_tracking = enable_body_tracking
        self.tracked_people = []
    
    def add_person_interactive(self, num_samples: int = 20, detector: str = "opencv"):
        """
        Interactive mode to add a new person to database
        
        Args:
            num_samples: Number of face samples to capture
            detector: Face detector to use
        """
        name = input("\nEnter person's name: ").strip()
        if not name:
            print("Invalid name!")
            return
        
        print(f"\n=== Adding {name} to Database ===")
        print(f"Capturing {num_samples} samples. Look at the camera and move your head slightly.")
        print("Press 'q' to stop early\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        samples_collected = 0
        frame_count = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.face_recognition.detect_faces(frame, detector=detector)
            
            for (x, y, w, h) in faces:
                if frame_count % 5 == 0:
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size > 0:
                        if self.face_recognition.add_person(name, face_img):
                            samples_collected += 1
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"Captured: {samples_collected}/{num_samples}", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            cv2.putText(frame, f"Embeddings: {samples_collected}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Adding Person - Press q to stop', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Added {samples_collected} face embeddings for {name}")
    
    def run_recognition(self, recognition_threshold: float = 0.6, detector: str = "opencv"):
        """
        Run real-time face recognition with continuous tracking
        
        Args:
            recognition_threshold: Recognition confidence threshold
            detector: Face detection backend
        """
        if self.face_recognition.collection.count() == 0:
            print("\n⚠ Warning: Database is empty! Add some people first.")
            input("Press Enter to continue anyway...")
        
        print("\n=== Face Recognition Started ===")
        print(f"Model: {self.face_recognition.model_name}")
        print(f"Detector: {detector}")
        print(f"Body Tracking: {'Enabled' if self.enable_body_tracking else 'Disabled'}")
        print(f"Recognition threshold: {recognition_threshold}")
        print("\n🎮 Controls:")
        print("  'q' - Quit")
        print("\n💡 Once recognized, person will be tracked continuously!")
        print()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # STEP 1: Update all existing trackers (every frame)
            trackers_to_remove = []
            for tracker in self.tracked_people:
                if not tracker.update_tracking(frame):
                    trackers_to_remove.append(tracker)
                    print(f"Lost track of {tracker.person_name}")
            
            for tracker in trackers_to_remove:
                self.tracked_people.remove(tracker)
            
            # STEP 2: Face detection and recognition (every 10 frames)
            if frame_count % 10 == 0:
                faces = self.face_recognition.detect_faces(frame, detector=detector)
                
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
                    
                    # Check if already tracked
                    matched_tracker = None
                    for tracker in self.tracked_people:
                        if tracker.overlaps_with_face((x, y, w, h)):
                            matched_tracker = tracker
                            break
                    
                    if matched_tracker:
                        # Update existing tracker
                        person_name, similarity = self.face_recognition.recognize_face(
                            face_img, threshold=recognition_threshold
                        )
                        if person_name == matched_tracker.person_name:
                            matched_tracker.update_with_face((x, y, w, h), frame, similarity)
                    else:
                        # New face - try to recognize
                        person_name, similarity = self.face_recognition.recognize_face(
                            face_img, threshold=recognition_threshold
                        )
                        
                        if person_name:
                            # Check if already tracking this person
                            already_tracked = any(t.person_name == person_name for t in self.tracked_people)
                            
                            if not already_tracked:
                                # Start tracking recognized person
                                new_tracker = PersonTracker(
                                    person_name, (x, y, w, h), frame, 
                                    similarity, self.enable_body_tracking
                                )
                                self.tracked_people.append(new_tracker)
                                print(f"Started tracking {person_name}")
                        else:
                            # Unknown person - just draw red box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 0, 255), -1)
                            cv2.putText(frame, "Unknown", (x+5, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # STEP 3: Draw all tracked people (every frame for smooth display)
            for tracker in self.tracked_people:
                x, y, w, h = tracker.get_box()
                
                # Draw thick green box for tracked person
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                
                # Draw label
                cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
                cv2.putText(frame, tracker.person_name, (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition with Tracking', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Face recognition ended")


def main():
    """Main menu for the application"""
    print("\n" + "="*60)
    print("    FACE RECOGNITION WITH CONTINUOUS TRACKING")
    print("="*60)
    print("\nAvailable Models:")
    print("1. Facenet512 (512-d, best accuracy) - RECOMMENDED")
    print("2. Facenet (128-d, fast and accurate)")
    print("3. ArcFace (512-d, state-of-the-art)")
    print("4. OpenFace (128-d, lightweight)")
    
    model_choice = input("\nSelect model (1-4, default 1): ").strip()
    model_map = {
        '1': 'Facenet512',
        '2': 'Facenet',
        '3': 'ArcFace',
        '4': 'OpenFace'
    }
    model_name = model_map.get(model_choice, 'Facenet512')
    
    print(f"\nInitializing with {model_name}...")
    app = FaceDetectionApp(model_name=model_name, enable_body_tracking=True)
    
    while True:
        print("\n" + "="*60)
        print(f"    FACE RECOGNITION SYSTEM ({model_name})")
        print("="*60)
        print("\n1. Add new person to database")
        print("2. Start face recognition with tracking")
        print("3. List all people in database")
        print("4. Delete person from database")
        print("5. Reset database (clear all data)")
        print("6. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            num_samples = input("Number of samples (default 20): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 20
            app.add_person_interactive(num_samples=num_samples)
        
        elif choice == '2':
            threshold = input("\nRecognition threshold (0.4-0.7, default 0.6): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.6
                threshold = max(0.0, min(1.0, threshold))
            except ValueError:
                threshold = 0.6
            app.run_recognition(recognition_threshold=threshold)
        
        elif choice == '3':
            app.face_recognition.list_people()
        
        elif choice == '4':
            name = input("\nEnter person's name to delete: ").strip()
            if name:
                app.face_recognition.delete_person(name)
        
        elif choice == '5':
            app.face_recognition.reset_database()
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice! Please try again.")


if __name__ == "__main__":
    main()

