# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, make_response
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import pandas as pd
import io
import datetime
import logging # For detailed logging

# Import your model and GradCAM utility
from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES # Import TARGET_PATHOLOGIES from model_utils
from utils.grad_cam import GradCAM
from utils.preprocessing import dicom_to_png, TARGET_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD # Import DICOM converter and constants

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'dcm', 'png', 'jpg', 'jpeg'} # Allow DICOM and common image formats
MAX_FILE_SIZE_MB = 50 # Max file size for uploads

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024 # Convert MB to bytes [11]
app.secret_key = os.urandom(24) # Generate a strong, random secret key for production [11, 27]

# Configure logging for the Flask app
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('app.log'),
                        logging.StreamHandler()
                    ])
app.logger.info("Flask app starting...")

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load your trained model and setup Grad-CAM once when the app starts
# Use CPU for inference if GPU is not available or for smaller deployments
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device for inference: {DEVICE}")

try:
    NUM_CLASSES = len(TARGET_PATHOLOGIES)
    MODEL_PATH = 'models/latest_model.pth' # Path to your trained model (e.g., the latest or best)
    
    # Adjust target_layer_name based on your chosen backbone and its internal structure
    # For ResNet/ResNeXt, 'base_model.layer4' is usually the last convolutional block [24, 25]
    # For EfficientNet, it might be 'base_model.features[-1]'
    TARGET_GRAD_CAM_LAYER = 'base_model.layer4' # Example for ResNet/ResNeXt

    model = MultiLabelResNet(num_classes=NUM_CLASSES, backbone='resnext101_32x8d', pretrained=False) # Set pretrained=False if loading your own trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode
    model.to(DEVICE)
    app.logger.info(f"Model loaded successfully from {MODEL_PATH}")

    grad_cam_explainer = GradCAM(model, TARGET_GRAD_CAM_LAYER)
    app.logger.info(f"Grad-CAM explainer initialized for layer: {TARGET_GRAD_CAM_LAYER}")
    
    # Define image preprocessing for inference (must match training preprocessing) [23, 2]
    inference_transform = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
except Exception as e:
    app.logger.error(f"Error loading model or Grad-CAM: {e}", exc_info=True)
    model = None # Indicate that model loading failed
    grad_cam_explainer = None
    flash(f"Server error: Model could not be loaded. Please contact support. Details: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_filepath)
            app.logger.info(f"File uploaded: {original_filepath}")
            
            # Generate a unique filename for processed image to avoid conflicts
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            base_filename_no_ext = os.path.splitext(filename)[0]
            processed_png_filename = f"{base_filename_no_ext}_{timestamp}.png"
            processed_png_filepath = os.path.join(app.config['RESULTS_FOLDER'], processed_png_filename) # Save processed image to results folder

            # Process the uploaded file
            if model and grad_cam_explainer:
                try:
                    # Convert DICOM to PNG if necessary
                    if filename.lower().endswith('.dcm'):
                        # For user uploads, we don't know the source dataset, so don't apply OpenI specific preprocessing
                        success = dicom_to_png(original_filepath, processed_png_filepath, apply_openi_specific_preprocessing=False)
                        if not success:
                            raise Exception("DICOM conversion failed.")
                        image_for_model_path = processed_png_filepath
                    else:
                        # For PNG/JPG, just copy to results folder and ensure it's RGB
                        img = Image.open(original_filepath).convert('RGB')
                        img.save(processed_png_filepath)
                        image_for_model_path = processed_png_filepath

                    # Preprocess image for model inference [23, 2]
                    image_for_model = Image.open(image_for_model_path).convert('RGB')
                    input_tensor = inference_transform(image_for_model).unsqueeze(0).to(DEVICE) # Add batch dim and move to device

                    # Perform inference
                    with torch.no_grad():
                        outputs = model(input_tensor) # Model outputs logits
                        probabilities = torch.sigmoid(outputs) # Apply sigmoid to get probabilities [1, 22]
                        
                    # Get predicted labels (threshold at 0.5)
                    predicted_labels_mask = (probabilities > 0.5).squeeze().cpu().numpy()
                    predicted_class_indices = np.where(predicted_labels_mask).tolist() # Ensure it's a list of indices
                    
                    predicted_class_names = [TARGET_PATHOLOGIES[idx] for idx in predicted_class_indices]
                    
                    # Get confidence scores for predicted classes
                    confidence_scores = {
                        TARGET_PATHOLOGIES[idx]: f"{probabilities.squeeze()[idx].item():.4f}"
                        for idx in predicted_class_indices
                    }
                    app.logger.info(f"Predictions for {filename}: {predicted_class_names} with scores {confidence_scores}")

                    # Generate and save Grad-CAM heatmaps for each predicted class [23, 24]
                    heatmap_filenames = []
                    for class_idx in predicted_class_indices:
                        # Clone tensor for each Grad-CAM call to avoid graph issues
                        heatmap = grad_cam_explainer.generate_heatmap(input_tensor.clone(), class_idx)
                        overlaid_img_np = grad_cam_explainer.overlay_heatmap(image_for_model_path, heatmap)
                        
                        heatmap_filename = f"heatmap_{TARGET_PATHOLOGIES[class_idx]}_{processed_png_filename}"
                        heatmap_filepath = os.path.join(app.config['RESULTS_FOLDER'], heatmap_filename)
                        cv2.imwrite(heatmap_filepath, cv2.cvtColor(overlaid_img_np, cv2.COLOR_RGB2BGR)) # Save as BGR for OpenCV
                        heatmap_filenames.append(heatmap_filename)
                    app.logger.info(f"Generated {len(heatmap_filenames)} heatmaps.")

                    # Generate a detailed report (simple NLG)
                    report_content = generate_detailed_report(predicted_class_names, confidence_scores, heatmap_filenames)
                    report_filename = f"report_{processed_png_filename.replace('.png', '.txt')}"
                    report_filepath = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
                    with open(report_filepath, 'w') as f:
                        f.write(report_content)
                    app.logger.info(f"Generated report: {report_filename}")

                    flash('File successfully uploaded and processed.')
                    return render_template('results.html', 
                                           original_image=processed_png_filename, # Display the processed PNG
                                           predictions=predicted_class_names,
                                           confidence_scores=confidence_scores,
                                           heatmaps=heatmap_filenames,
                                           report_filename=report_filename)

                except Exception as e:
                    flash(f"Error during image processing or prediction: {e}")
                    app.logger.error(f"Processing error for {filename}: {e}", exc_info=True) # Log detailed error
                    return redirect(request.url)
            else:
                flash("Model not loaded. Please check server logs for details.")
                app.logger.error("Attempted to process file but model was not loaded.")
                return redirect(request.url)
        else:
            flash(f'Allowed image types are {", ".join(ALLOWED_EXTENSIONS)}. Max file size is {MAX_FILE_SIZE_MB}MB.')
            app.logger.warning(f"Invalid file upload attempt: {filename}")
            return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve original uploaded files (e.g., DICOM if needed for raw view) [11, 28, 29]
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    # Serve processed images and heatmaps [11, 28, 29]
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/download_report/<filename>')
def download_report(filename):
    """Allows downloading the generated text report."""
    report_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(report_path):
        return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)
    else:
        flash("Report not found.")
        return redirect(url_for('index'))

def generate_detailed_report(predicted_class_names, confidence_scores, heatmap_filenames):
    """
    Generates a detailed text report based on model predictions and XAI insights.
    This is a simplified Natural Language Generation (NLG) example.
    For MedGemma-level reports, a dedicated LLM fine-tuned on radiology reports would be needed.
    """
    report = f"--- Chest X-ray Analysis Report ---\n"
    report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**Predicted Findings:**\n"

    if not predicted_class_names:
        report += "  - No specific pathologies detected with high confidence.\n"
    else:
        for class_name in predicted_class_names:
            confidence = confidence_scores.get(class_name, 'N/A')
            report += f"  - {class_name}: Confidence Score = {confidence}\n"
            # Add a simple explanation for why it chose the output
            report += f"    * Explanation: The model identified visual patterns consistent with {class_name} in the X-ray image. These patterns often include specific opacities, consolidations, or changes in organ size/shape. The confidence score indicates the model's certainty in this prediction.\n"
            # In a real system, this explanation would be more nuanced, potentially linking to specific features.

    report += "\n**Visual Explanations (Grad-CAM Heatmaps):**\n"
    report += "  Heatmaps highlight the specific regions of the X-ray image that most influenced the model's decision for each predicted finding. Red areas indicate higher importance.\n"
    if heatmap_filenames:
        for h_filename in heatmap_filenames:
            class_name_from_heatmap = h_filename.split('_')[1] # Extract class name from filename
            report += f"    * {class_name_from_heatmap} Heatmap: Refer to the corresponding heatmap image for visual localization of findings.\n"
    else:
        report += "  - No heatmaps generated (no significant findings or an issue occurred during generation).\n"

    report += "\n**Disclaimer:**\n"
    report += "This report is generated by an AI model and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any medical concerns. AI models, while powerful, are tools to assist and not replace human expertise.\n"
    report += "--- End of Report ---\n"
    return report

if __name__ == '__main__':
    # For local testing, set debug=True. For production, set debug=False.
    # When deploying with Gunicorn/Nginx, this block is not directly executed.
    app.run(debug=True, host='0.0.0.0', port=5000)