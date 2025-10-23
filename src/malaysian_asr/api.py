"""
FastAPI server for Malaysian ASR transcription service.
"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .core.transcriber import MERaLiONTranscriber


# Pydantic models for API responses
class TaskResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    message: str


class ProgressResponse(BaseModel):
    current_task: Optional[Dict]
    queue_length: int
    completed_tasks: int
    failed_tasks: int
    total_tasks: int


class TaskStatus(BaseModel):
    task_id: str
    filename: str
    status: str  # "queued", "processing", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None


# Global task management
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskStatus] = {}
        self.current_task_id: Optional[str] = None
        self.completed_count = 0
        self.failed_count = 0
        self.transcriber: Optional[MERaLiONTranscriber] = None
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def initialize_transcriber(self):
        """Initialize the transcriber if not already done."""
        if self.transcriber is None:
            self.transcriber = MERaLiONTranscriber()
            if not self.transcriber.load_model():
                raise Exception("Failed to load MERaLiON model")
    
    def add_task(self, filename: str) -> str:
        """Add a new transcription task."""
        task_id = str(uuid.uuid4())
        task = TaskStatus(
            task_id=task_id,
            filename=filename,
            status="queued",
            created_at=datetime.now()
        )
        self.tasks[task_id] = task
        return task_id
    
    def start_task(self, task_id: str):
        """Mark a task as processing."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "processing"
            self.tasks[task_id].started_at = datetime.now()
            self.current_task_id = task_id
    
    def complete_task(self, task_id: str, result: str):
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].completed_at = datetime.now()
            self.tasks[task_id].result = result
            self.completed_count += 1
            self.current_task_id = None
    
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].completed_at = datetime.now()
            self.tasks[task_id].error = error
            self.failed_count += 1
            self.current_task_id = None
    
    def get_progress(self) -> ProgressResponse:
        """Get current progress information."""
        current_task = None
        if self.current_task_id and self.current_task_id in self.tasks:
            task = self.tasks[self.current_task_id]
            current_task = {
                "task_id": task.task_id,
                "filename": task.filename,
                "status": task.status,
                "started_at": task.started_at.isoformat() if task.started_at else None
            }
        
        queued_tasks = sum(1 for task in self.tasks.values() if task.status == "queued")
        
        return ProgressResponse(
            current_task=current_task,
            queue_length=queued_tasks,
            completed_tasks=self.completed_count,
            failed_tasks=self.failed_count,
            total_tasks=len(self.tasks)
        )


# Global task manager instance
task_manager = TaskManager()

# FastAPI app
app = FastAPI(
    title="Malaysian ASR API",
    description="FastAPI server for MERaLiON speech transcription",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize the transcriber on startup."""
    try:
        task_manager.initialize_transcriber()
        print("✅ MERaLiON transcriber initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize transcriber: {e}")


async def process_transcription_task(task_id: str, file_path: str):
    """Background task to process transcription."""
    try:
        task_manager.start_task(task_id)
        
        # Perform transcription
        result = task_manager.transcriber.transcribe(file_path)
        
        if result:
            task_manager.complete_task(task_id, result)
            print(f"✅ Task {task_id} completed: {task_manager.tasks[task_id].filename}")
        else:
            task_manager.fail_task(task_id, "Transcription returned empty result")
            print(f"❌ Task {task_id} failed: Empty result")
            
    except Exception as e:
        task_manager.fail_task(task_id, str(e))
        print(f"❌ Task {task_id} failed: {e}")
    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass


@app.post("/upload", response_model=TaskResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a WAV file for transcription.
    
    Returns:
        TaskResponse with task_id and status
    """
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(
            status_code=400,
            detail="Only audio files (.wav, .mp3, .flac, .m4a) are supported"
        )
    
    # Check if transcriber is available
    if task_manager.transcriber is None:
        raise HTTPException(
            status_code=503,
            detail="Transcription service not available. Please try again later."
        )
    
    try:
        # Save uploaded file
        file_path = task_manager.upload_dir / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add task to queue
        task_id = task_manager.add_task(file.filename)
        
        # Start background processing
        background_tasks.add_task(process_transcription_task, task_id, str(file_path))
        
        return TaskResponse(
            task_id=task_id,
            filename=file.filename,
            status="queued",
            message=f"File uploaded successfully. Task ID: {task_id}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@app.get("/progress", response_model=ProgressResponse)
async def get_progress():
    """
    Get current transcription progress and queue status.
    
    Returns:
        ProgressResponse with current task, queue length, and statistics
    """
    return task_manager.get_progress()


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get detailed status of a specific task.
    
    Returns:
        TaskStatus with full task details
    """
    if task_id not in task_manager.tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    return task_manager.tasks[task_id]


@app.get("/tasks")
async def list_tasks():
    """
    List all tasks with their status.
    
    Returns:
        List of all tasks
    """
    return list(task_manager.tasks.values())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    transcriber_status = "available" if task_manager.transcriber else "unavailable"
    return {
        "status": "healthy",
        "transcriber": transcriber_status,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
