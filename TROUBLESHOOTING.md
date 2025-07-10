# Troubleshooting: "Failed to fetch" Error

## Problem
You're getting a "Upload failed: TypeError: Failed to fetch" error when trying to upload audio files.

## Solution

### Step 1: Start the Backend Server

The "Failed to fetch" error occurs because the frontend can't connect to the backend server. You need to start the backend server first.

#### Option A: Use the Simplified Server (Recommended)
```bash
cd backend
python start_server.py
```

#### Option B: Manual Start
```bash
cd backend
python api_simple.py
```

### Step 2: Test the Connection

1. Open `frontend/test.html` in your browser
2. Click "Test Backend Connection"
3. If you see "Success: {"message": "Backend is running!"}", the server is working

### Step 3: Check Dependencies

If the server won't start, you may need to install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask flask-cors openai-whisper nltk torch numpy
```

### Step 4: Verify Server is Running

The server should be running on `http://127.0.0.1:5000`

You should see output like:
```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://[::1]:5000
```

### Step 5: Test Upload

1. Make sure the backend server is running
2. Open `frontend/upload.html` in your browser
3. Try uploading an audio file
4. You should see the transcription results

## Common Issues

### Issue 1: Port 5000 is already in use
**Solution**: Kill the process using port 5000 or change the port in `api_simple.py`

### Issue 2: CORS errors
**Solution**: The simplified server includes CORS headers. If you still get CORS errors, make sure you're using `api_simple.py`

### Issue 3: Whisper model download issues
**Solution**: The first time you run the server, it will download the Whisper model. This may take a few minutes.

### Issue 4: Memory issues with large audio files
**Solution**: The simplified server processes audio synchronously. For very large files, consider using the original Redis-based system.

## What Changed

I've created a simplified version of the backend (`api_simple.py`) that:
- Doesn't require Redis
- Processes audio synchronously (no background jobs)
- Returns results immediately
- Includes better error handling

The frontend has been updated to work with this simplified backend.

## Original vs Simplified System

| Feature | Original (api.py) | Simplified (api_simple.py) |
|---------|------------------|---------------------------|
| Background processing | ✅ (Redis/RQ) | ❌ (Synchronous) |
| Large file support | ✅ | ⚠️ (Limited) |
| Dependencies | Redis, RQ | Flask, Whisper, NLTK |
| Setup complexity | High | Low |
| Error handling | Basic | Improved |

Use the simplified system for testing and development. Use the original system for production with large files. 