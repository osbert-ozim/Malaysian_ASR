"""
FastAPI server for Malaysian ASR transcription service.
"""

import os
import uuid
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
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
        self.results_dir = Path("./results")
        self.tasks_file = Path("./tasks.json")
        
        # Create directories
        self.upload_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load existing tasks
        self.load_tasks()
    
    def load_tasks(self):
        """Load tasks from persistent storage."""
        if self.tasks_file.exists():
            try:
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for task_id, task_data in data.items():
                        # Convert datetime strings back to datetime objects
                        if task_data.get('created_at'):
                            task_data['created_at'] = datetime.fromisoformat(task_data['created_at'])
                        if task_data.get('started_at'):
                            task_data['started_at'] = datetime.fromisoformat(task_data['started_at'])
                        if task_data.get('completed_at'):
                            task_data['completed_at'] = datetime.fromisoformat(task_data['completed_at'])
                        
                        self.tasks[task_id] = TaskStatus(**task_data)
                        
                # Update counters
                self.completed_count = sum(1 for task in self.tasks.values() if task.status == "completed")
                self.failed_count = sum(1 for task in self.tasks.values() if task.status == "failed")
                
                # Set current task if there's one processing
                for task_id, task in self.tasks.items():
                    if task.status == "processing":
                        self.current_task_id = task_id
                        break
                        
                print(f"✅ Loaded {len(self.tasks)} tasks from storage")
            except Exception as e:
                print(f"⚠️  Failed to load tasks: {e}")
    
    def save_tasks(self):
        """Save tasks to persistent storage."""
        try:
            data = {}
            for task_id, task in self.tasks.items():
                task_dict = task.dict()
                # Convert datetime objects to ISO strings
                if task_dict.get('created_at'):
                    task_dict['created_at'] = task_dict['created_at'].isoformat()
                if task_dict.get('started_at'):
                    task_dict['started_at'] = task_dict['started_at'].isoformat()
                if task_dict.get('completed_at'):
                    task_dict['completed_at'] = task_dict['completed_at'].isoformat()
                data[task_id] = task_dict
            
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Failed to save tasks: {e}")
    
    def save_result(self, task_id: str, result: str):
        """Save transcription result to file."""
        try:
            result_file = self.results_dir / f"{task_id}.txt"
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"✅ Result saved to {result_file}")
        except Exception as e:
            print(f"⚠️  Failed to save result: {e}")
    
    def load_result(self, task_id: str) -> Optional[str]:
        """Load transcription result from file."""
        try:
            result_file = self.results_dir / f"{task_id}.txt"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"⚠️  Failed to load result: {e}")
        return None
    
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
        self.save_tasks()
        return task_id
    
    def start_task(self, task_id: str):
        """Mark a task as processing."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "processing"
            self.tasks[task_id].started_at = datetime.now()
            self.current_task_id = task_id
            self.save_tasks()
    
    def complete_task(self, task_id: str, result: str):
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].completed_at = datetime.now()
            self.tasks[task_id].result = result
            self.completed_count += 1
            self.current_task_id = None
            
            # Save result to file
            self.save_result(task_id, result)
            self.save_tasks()
    
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].completed_at = datetime.now()
            self.tasks[task_id].error = error
            self.failed_count += 1
            self.current_task_id = None
            self.save_tasks()
    
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


@app.get("/task-ids")
async def list_task_ids():
    """
    List all task IDs with basic information.
    
    Returns:
        List of task IDs with filename and status
    """
    task_list = []
    for task_id, task in task_manager.tasks.items():
        task_list.append({
            "task_id": task_id,
            "filename": task.filename,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        })
    
    # Sort by creation time (newest first)
    task_list.sort(key=lambda x: x["created_at"], reverse=True)
    return task_list


@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """
    Get the transcription result for a specific task.
    
    Returns:
        Transcription result as text/plain or JSON
    """
    if task_id not in task_manager.tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    task = task_manager.tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task.status}"
        )
    
    # Try to load result from file first, fallback to in-memory
    result = task_manager.load_result(task_id)
    if result is None:
        result = task.result
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Result not found"
        )
    
    return {
        "task_id": task_id,
        "filename": task.filename,
        "result": result,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None
    }


@app.get("/tasks/{task_id}/result/download")
async def download_task_result(task_id: str):
    """
    Download the transcription result as a text file.
    
    Returns:
        Text file with transcription result
    """
    if task_id not in task_manager.tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    task = task_manager.tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task.status}"
        )
    
    # Try to load result from file first, fallback to in-memory
    result = task_manager.load_result(task_id)
    if result is None:
        result = task.result
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Result not found"
        )
    
    # Create a temporary file for download
    result_file = task_manager.results_dir / f"{task_id}_download.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcription Result for: {task.filename}\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Completed: {task.completed_at.isoformat() if task.completed_at else 'Unknown'}\n")
        f.write("=" * 50 + "\n\n")
        f.write(result)
    
    return FileResponse(
        path=str(result_file),
        filename=f"{task.filename}_transcript.txt",
        media_type="text/plain"
    )


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
    uvicorn.run(app, host="0.0.0.0", port=54321)
