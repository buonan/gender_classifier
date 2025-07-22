import os
import torch
from torchvision import transforms, models
from PIL import Image
import io
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
def load_model():
    # Check if model exists
    model_path = 'models/gender_classifier.pth'
    if not os.path.exists(model_path):
        return None, None
    
    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: male and female
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint.get('class_names', ['male', 'female'])
    
    return model, class_names

model, class_names = load_model()

# Image transformation for prediction
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read and process the image
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            prediction = class_names[preds.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item()
            
        # Convert image to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence * 100:.2f}%",
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/model-status')
def model_status():
    if model is None:
        return jsonify({'status': 'not_loaded'})
    return jsonify({'status': 'loaded'})

if __name__ == '__main__':
    app.run(debug=True)
