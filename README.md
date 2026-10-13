# YOLOv8 Image Tester (Streamlit)

A simple Streamlit app for testing a trained Ultralytics YOLOv8 model on uploaded images.

## Files you need

- `app.py` — web interface and inference code
- `requirements.txt` — Python dependencies
- `best.pt` — **your trained model file** (you must add this yourself)

## Run locally

1. Put your model file in this folder and name it `best.pt`.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and this `README.md`.
3. Add `best.pt` to the repository if it is within GitHub's file-size limits.
4. On Streamlit Community Cloud, choose **Deploy an app from GitHub**.
5. Select your repository, branch, and `app.py`, then deploy.

### Important: large model files

GitHub blocks regular files larger than 100 MB. If `best.pt` is too large, do not try to commit it as a normal file. Store it in a suitable model-hosting service (for example, Hugging Face Hub) and modify `app.py` to download it securely at startup, or use Git LFS where supported. Streamlit deployments also have memory and resource limits.

## What the app does

- Accepts JPG, JPEG, PNG, WEBP, and BMP images
- Runs model inference with an adjustable confidence threshold and image size
- Displays bounding boxes, class labels, and confidence scores
- Shows a detection table and offers CSV export

## Model compatibility

This app expects an Ultralytics-compatible YOLO model checkpoint. It detects the classes your model was trained to recognize; a custom wildlife model will not automatically detect every general-purpose object.
