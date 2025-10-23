#!/usr/bin/env python3
"""
Enhanced test script for Malaysian ASR API endpoints.
"""

import requests
import time
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_api():
    """Test the enhanced API endpoints."""
    print("🧪 Testing Enhanced Malaysian ASR API")
    print("=" * 60)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running with 'make api'")
        return
    
    # Test task IDs endpoint (should be empty initially)
    print("\n2. Testing task IDs endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/task-ids")
        if response.status_code == 200:
            task_ids = response.json()
            print("✅ Task IDs endpoint working")
            print(f"   Found {len(task_ids)} tasks")
            if task_ids:
                print("   Recent tasks:")
                for task in task_ids[:3]:  # Show first 3
                    print(f"     - {task['task_id'][:8]}... ({task['filename']}) - {task['status']}")
        else:
            print(f"❌ Task IDs endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Task IDs endpoint error: {e}")
    
    # Test progress endpoint
    print("\n3. Testing progress endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/progress")
        if response.status_code == 200:
            progress = response.json()
            print("✅ Progress endpoint working")
            print(f"   Queue length: {progress['queue_length']}")
            print(f"   Completed tasks: {progress['completed_tasks']}")
            print(f"   Failed tasks: {progress['failed_tasks']}")
            print(f"   Total tasks: {progress['total_tasks']}")
        else:
            print(f"❌ Progress endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Progress endpoint error: {e}")
    
    # Test file upload (if test file exists)
    test_file = Path("src/malaysian_asr/data/test_sample.wav")
    if test_file.exists():
        print(f"\n4. Testing file upload with {test_file.name}...")
        try:
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "audio/wav")}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ File upload successful")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Filename: {result['filename']}")
                print(f"   Status: {result['status']}")
                
                task_id = result['task_id']
                
                # Monitor progress
                print(f"\n5. Monitoring task {task_id}...")
                for i in range(30):  # Monitor for up to 30 seconds
                    time.sleep(1)
                    progress_response = requests.get(f"{API_BASE_URL}/progress")
                    if progress_response.status_code == 200:
                        progress = progress_response.json()
                        print(f"   Progress: Queue={progress['queue_length']}, "
                              f"Completed={progress['completed_tasks']}, "
                              f"Failed={progress['failed_tasks']}")
                        
                        if progress['current_task'] is None and progress['queue_length'] == 0:
                            print("✅ Task completed!")
                            break
                    else:
                        print(f"❌ Progress check failed: {progress_response.status_code}")
                        break
                else:
                    print("⏰ Monitoring timeout")
                
                # Test task details endpoint
                print(f"\n6. Testing task details for {task_id}...")
                task_response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")
                if task_response.status_code == 200:
                    task_details = task_response.json()
                    print("✅ Task details retrieved")
                    print(f"   Status: {task_details['status']}")
                    if task_details['result']:
                        print(f"   Result preview: {task_details['result'][:100]}...")
                    if task_details['error']:
                        print(f"   Error: {task_details['error']}")
                else:
                    print(f"❌ Task details failed: {task_response.status_code}")
                
                # Test result endpoint (if task completed)
                if task_response.status_code == 200:
                    task_details = task_response.json()
                    if task_details['status'] == 'completed':
                        print(f"\n7. Testing result endpoint for {task_id}...")
                        result_response = requests.get(f"{API_BASE_URL}/tasks/{task_id}/result")
                        if result_response.status_code == 200:
                            result_data = result_response.json()
                            print("✅ Result endpoint working")
                            print(f"   Filename: {result_data['filename']}")
                            print(f"   Result preview: {result_data['result'][:100]}...")
                            print(f"   Completed at: {result_data['completed_at']}")
                        else:
                            print(f"❌ Result endpoint failed: {result_response.status_code}")
                        
                        # Test download endpoint
                        print(f"\n8. Testing download endpoint for {task_id}...")
                        download_response = requests.get(f"{API_BASE_URL}/tasks/{task_id}/result/download")
                        if download_response.status_code == 200:
                            print("✅ Download endpoint working")
                            print(f"   Content-Type: {download_response.headers.get('content-type')}")
                            print(f"   Content-Length: {len(download_response.content)} bytes")
                        else:
                            print(f"❌ Download endpoint failed: {download_response.status_code}")
                    else:
                        print(f"\n7. Skipping result tests (task status: {task_details['status']})")
                        
            else:
                print(f"❌ File upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ File upload error: {e}")
    else:
        print(f"\n4. Skipping file upload test (test file not found: {test_file})")
    
    # Test task IDs endpoint again (should show the uploaded task)
    print(f"\n9. Testing task IDs endpoint again...")
    try:
        response = requests.get(f"{API_BASE_URL}/task-ids")
        if response.status_code == 200:
            task_ids = response.json()
            print("✅ Task IDs endpoint working")
            print(f"   Found {len(task_ids)} tasks")
            if task_ids:
                print("   Recent tasks:")
                for task in task_ids[:5]:  # Show first 5
                    print(f"     - {task['task_id'][:8]}... ({task['filename']}) - {task['status']}")
        else:
            print(f"❌ Task IDs endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Task IDs endpoint error: {e}")
    
    print(f"\n{'=' * 60}")
    print("🏁 Enhanced API test completed!")
    print("\n💡 New endpoints available:")
    print(f"   - List task IDs: GET {API_BASE_URL}/task-ids")
    print(f"   - Get task result: GET {API_BASE_URL}/tasks/{{task_id}}/result")
    print(f"   - Download result: GET {API_BASE_URL}/tasks/{{task_id}}/result/download")
    print(f"   - View API docs: {API_BASE_URL}/docs")

if __name__ == "__main__":
    test_api()
