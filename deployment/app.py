import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- 1. LOAD THE TRAINED MODEL ---
# We must define the model architecture exactly as it was during training
def load_model():
    print("Loading model...")
    # Initialize ResNet18 architecture
    model = models.resnet18(weights=None) 
    # Match the output layer to 2 classes (Normal vs Pneumonia)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load the weights you trained
    # map_location='cpu' ensures it runs on your laptop even if trained on GPU
    model.load_state_dict(torch.load('pneumonia_resnet18.pth', map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
    return model

model = load_model()

# --- 2. IMAGE PREPROCESSING ---
# This must match the Validation transform from your training code EXACTLY
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)

# --- 3. API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Process image
        tensor = transform_image(file.read())
        
        # 2. Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Map index to class name
            class_id = predicted.item()
            class_name = "PNEUMONIA" if class_id == 1 else "NORMAL"
            confidence_score = confidence[class_id].item()

        # 3. Return JSON
        return jsonify({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': f"{confidence_score:.2%}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)