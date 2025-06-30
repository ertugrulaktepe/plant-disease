from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io
import numpy as np
import os

app = Flask(__name__)




def load_model():
    model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    local_dir = "./plant_disease_model"

    if not os.path.exists(model_name):
        print("Model indiriliyor...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
        )
        print("Model indirildi!")
    

    processor = AutoImageProcessor.from_pretrained(local_dir)
    model = AutoModelForImageClassification.from_pretrained(local_dir)
    return model, processor


def preprocess_image(image, processor):
    try:
        
        img = Image.open(io.BytesIO(image))
        
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        
        print(f"Görüntü boyutları: {img.size}")
        print(f"Görüntü modu: {img.mode}")
        
        
        inputs = processor(images=img, return_tensors="pt")
        return inputs
    except Exception as e:
        print(f"Görüntü işleme hatası: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Görüntü yüklenmedi'}), 400
    
    try:
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            print(f"Yüklenen görüntü formatı: {img.format}")
            print(f"Yüklenen görüntü boyutları: {img.size}")
            print(f"Yüklenen görüntü modu: {img.mode}")
        except Exception as e:
            return jsonify({'error': f'Geçersiz görüntü formatı: {str(e)}'}), 400
        
        
        model, processor = load_model()
        
        
        inputs = preprocess_image(image_bytes, processor)
        
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.softmax(dim=1)
            predicted_class = torch.argmax(predictions).item()
            confidence = float(predictions[0][predicted_class])
        
        
        class_name = model.config.id2label[predicted_class]
        
        result = {
            "disease": class_name,
            "confidence": confidence,
            "model": "MobileNetV2 Plant Disease Identification"
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
