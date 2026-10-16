# app.py
from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import threading
import zipfile
import json
import subprocess
import cv2

# Modular imports
from core.scene_detector import run_scene_detection, get_ffmpeg_path
from core.motion_analyzer import analyze_motion
from core.image_stitcher import stitch_frames
from core.semantic_filter import SemanticFilter, is_valid_presentation
from utils.progress_tracker import update_progress
from utils.file_helper import get_extracted_images, cleanup_uploaded_video
import config

app = Flask(__name__)

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

def bg_extraction_worker(task_id, video_path, output_dir, crop_x, crop_y, crop_w, crop_h):
    try:
        # 1. Run Scene Detection
        timestamps = run_scene_detection(
            video_path, output_dir, crop_x, crop_y, crop_w, crop_h,
            threshold=config.DEFAULT_SCENE_THRESHOLD
        )
        
        update_progress(output_dir, 50, f"Detected {len(timestamps)} scenes. Extracting frames...")
        
        # 2. Extract and crop images to output dir
        ffmpeg_path = get_ffmpeg_path()
        total_timestamps = len(timestamps)
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        for i, t in enumerate(timestamps):
            out_file = os.path.join(output_dir, f"frame_{i+1:04d}.png")
            command = [
                ffmpeg_path,
                "-ss", f"{t:.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-y",
                out_file
            ]
            
            try:
                subprocess.run(command, check=True, startupinfo=startupinfo)
                
                # Physical ROI cropping
                if crop_x != 0 or crop_y != 0 or crop_w != 100 or crop_h != 100:
                    img = cv2.imread(out_file)
                    if img is not None:
                        h, w = img.shape[:2]
                        x1 = int(w * (crop_x / 100.0))
                        y1 = int(h * (crop_y / 100.0))
                        x2 = int(w * ((crop_x + crop_w) / 100.0))
                        y2 = int(h * ((crop_y + crop_h) / 100.0))
                        
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(x1 + 1, min(x2, w))
                        y2 = max(y1 + 1, min(y2, h))
                        
                        cropped_img = img[y1:y2, x1:x2]
                        cv2.imwrite(out_file, cropped_img)
            except Exception as e:
                print(f"Extraction or cropping error: {e}")
                
            percent = 50 + int(((i + 1) / total_timestamps) * 40)
            update_progress(output_dir, percent, f"Extracting frame {i+1} of {total_timestamps}...")
            
        # 3. AI Post-Processing (Scroll Stitching & Semantic Similarity De-duplication)
        update_progress(output_dir, 92, "Initializing AI models & analyzing screenshots...")
        
        ai_filter = SemanticFilter()
        images = get_extracted_images(output_dir)
        
        # Clean up non-presentation frames (webcam grids, lobbies, dark idle avatars)
        valid_images = []
        for img_name in images:
            img_path = os.path.join(output_dir, img_name)
            if not is_valid_presentation(img_path):
                try:
                    os.remove(img_path)
                except Exception as ex:
                    print(f"Failed to delete non-presentation frame {img_name}: {ex}")
            else:
                valid_images.append(img_name)
                
        images = valid_images
        
        if len(images) >= 2:
            pruned = 0
            stitched = 0
            last_active_img = images[0]
            
            for idx in range(1, len(images)):
                curr_img_name = images[idx]
                path1 = os.path.join(output_dir, last_active_img)
                path2 = os.path.join(output_dir, curr_img_name)
                
                if not os.path.exists(path1) or not os.path.exists(path2):
                    continue
                    
                img1_gray = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
                img2_gray = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
                
                if img1_gray is None or img2_gray is None:
                    continue
                    
                # RANSAC Motion Analyzer
                is_scroll, dy, is_static_duplicate = analyze_motion(img1_gray, img2_gray)
                
                if is_scroll:
                    # Stitch scroll band onto baseline canvas
                    success = stitch_frames(path1, path2, dy)
                    if success:
                        os.remove(path2)
                        stitched += 1
                        pruned += 1
                        continue
                elif is_static_duplicate:
                    # If motion analyzer says it is a static duplicate (no motion), delete it!
                    os.remove(path2)
                    pruned += 1
                    continue
                        
                # Semantic Similarity evaluation (CNN Model or Fallback Correlation)
                similarity = ai_filter.compute_similarity(path1, path2)
                
                if similarity >= 0.99:  # Threshold for layout similarity
                    os.remove(path2)
                    pruned += 1
                else:
                    last_active_img = curr_img_name
                    
            msg = f"Completed! Saved {len(get_extracted_images(output_dir))} screens"
            details = []
            if stitched > 0:
                details.append(f"stitched {stitched} scroll events")
            if pruned - stitched > 0:
                details.append(f"pruned {pruned - stitched} duplicate frames")
            if details:
                msg += f" ({', '.join(details)})."
            else:
                msg += "."
                
            update_progress(output_dir, 100, msg, status="completed")
        else:
            update_progress(output_dir, 100, f"Extraction complete! Saved {len(images)} screenshots.", status="completed")
            
    except Exception as e:
        print(f"Error in background worker for task {task_id}: {e}")
        update_progress(output_dir, 100, f"Error: {str(e)}", status="failed", error=str(e))
    finally:
        cleanup_uploaded_video(video_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
        
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
        
    try:
        crop_x = int(request.form.get("crop_x", 0))
        crop_y = int(request.form.get("crop_y", 0))
        crop_w = int(request.form.get("crop_w", 100))
        crop_h = int(request.form.get("crop_h", 100))
    except ValueError:
        return jsonify({"error": "Invalid parameter values"}), 400

    task_id = str(uuid.uuid4())
    task_upload_dir = os.path.join(config.UPLOAD_FOLDER, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)
    
    file_ext = os.path.splitext(file.filename)[1]
    video_path = os.path.join(task_upload_dir, f"video{file_ext}")
    file.save(video_path)
    
    output_dir = os.path.join(config.OUTPUT_FOLDER, task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    update_progress(output_dir, 0, "Initializing background job...", status="pending")
    
    thread = threading.Thread(
        target=bg_extraction_worker,
        args=(task_id, video_path, output_dir, crop_x, crop_y, crop_w, crop_h)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route("/status/<task_id>", methods=["GET"])
def status(task_id):
    output_dir = os.path.join(config.OUTPUT_FOLDER, task_id)
    progress_file = os.path.join(output_dir, "progress.json")
    
    if not os.path.exists(progress_file):
        return jsonify({"status": "unknown", "progress": 0, "message": "Task not found."}), 404
        
    try:
        with open(progress_file, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"status": "error", "progress": 0, "message": f"Error reading state: {str(e)}"}), 500

@app.route("/delete_frame/<task_id>/<filename>", methods=["POST"])
def delete_frame(task_id, filename):
    output_dir = os.path.join(config.OUTPUT_FOLDER, task_id)
    file_path = os.path.join(output_dir, filename)
    
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
        
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            
            progress_file = os.path.join(output_dir, "progress.json")
            if os.path.exists(progress_file):
                with open(progress_file, "r") as f:
                    data = json.load(f)
                
                data["images"] = get_extracted_images(output_dir)
                
                with open(progress_file, "w") as f:
                    json.dump(data, f, indent=2)
                    
            return jsonify({"status": "success", "images": get_extracted_images(output_dir)})
        except Exception as e:
            return jsonify({"error": f"Failed to delete frame: {str(e)}"}), 500
    else:
        return jsonify({"error": "File not found"}), 404

@app.route("/download_zip/<task_id>", methods=["GET"])
def download_zip(task_id):
    output_dir = os.path.join(config.OUTPUT_FOLDER, task_id)
    
    if not os.path.exists(output_dir):
        return "Task output not found", 404
        
    zip_path = os.path.join(output_dir, "extracted_screens.zip")
    
    try:
        images = get_extracted_images(output_dir)
        if not images:
            return "No images found to package", 400
            
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img in images:
                full_img_path = os.path.join(output_dir, img)
                if os.path.exists(full_img_path):
                    zipf.write(full_img_path, img)
                    
        return send_file(zip_path, as_attachment=True, download_name="extracted_screens.zip")
    except Exception as e:
        return f"Error creating ZIP: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)