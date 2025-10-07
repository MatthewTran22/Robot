# Voice Recognition & Transcription System

Real-time speech-to-text transcription with speaker identification.

## Features

- **Speech-to-Text**: Converts spoken words to text using Faster-Whisper
- **Speaker Recognition**: Identifies who is speaking using voice embeddings (Resemblyzer)
- **Real-time Processing**: Live audio processing with configurable chunk duration
- **Output Format**: `[timestamp] Name (confidence): message`

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Enroll Speakers

Before using speaker recognition, you need to enroll speakers by recording their voices:

```bash
python Transcribe/enroll_voice.py
```

Options:
- **Option 1**: Record voice directly (recommended - records 5 seconds of audio)
- **Option 2**: Use existing audio file
- **Option 3**: Remove enrolled speaker
- **Option 4**: List all enrolled speakers

**Tips for best enrollment:**
- Speak clearly for 5-10 seconds
- Say varied sentences (not just one word)
- Use a quiet environment
- Use the same microphone you'll use for transcription

### Step 2: Run Transcription

Start real-time transcription with speaker recognition:

```bash
python Transcribe/Transcriber.py
```

You'll be prompted to:
1. Select Whisper model size (base recommended)
2. Choose language (or auto-detect)
3. Set chunk duration (5 seconds recommended)
4. Enable/disable speaker recognition

### Output Example

```
[14:32:15] Matthew (85%): Hello, how are you doing today?
[14:32:18] Sarah (92%): I'm doing great, thanks for asking!
[14:32:22] Unknown (45%): [unintelligible or unrecognized speaker]
```

## File Structure

```
Transcribe/
├── Transcriber.py          # Main transcription module
├── VoiceRecognition.py     # Speaker recognition module  
├── enroll_voice.py         # Voice enrollment tool
└── voice_database.pkl      # Saved voice profiles (auto-created)
```

## How It Works

1. **Voice Enrollment**: 
   - Records audio sample
   - Extracts voice embedding (128-D vector representing speaker characteristics)
   - Saves to database

2. **Speaker Identification**:
   - Processes audio chunk
   - Extracts voice embedding
   - Compares with enrolled profiles using cosine similarity
   - Returns best match if confidence > threshold (75%)

3. **Transcription**:
   - Converts audio to text using Whisper model
   - Combines speaker name + transcribed text
   - Outputs as: `Name: message`

## Configuration

### Transcriber Options
- `model_size`: tiny, base, small, medium, large-v3
- `enable_speaker_recognition`: True/False
- `chunk_duration`: Audio processing interval (seconds)

### Voice Recognition Options
- `threshold`: Similarity threshold for recognition (0-1, default 0.75)
  - Higher = stricter matching (fewer false positives)
  - Lower = looser matching (may recognize similar voices)

## Troubleshooting

**No speakers enrolled:**
- Run `enroll_voice.py` first to register speakers

**Poor recognition accuracy:**
- Re-enroll with longer audio samples (10+ seconds)
- Ensure good audio quality
- Increase threshold for stricter matching

**Speaker shows as "Unknown":**
- Confidence below threshold
- Voice not enrolled in database
- Poor audio quality or background noise

## Requirements

- Python 3.8+
- Microphone access
- ~500MB for Whisper model download (first run)
- ~100MB RAM per enrolled speaker

