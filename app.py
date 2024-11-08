import torch
import numpy as np
from flask import Flask, request, jsonify
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class SimpleNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sizes[1], output_size),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

class ModelInference:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['inference']['device'])
        self.model = self._load_model()
        self.threshold = config['inference']['threshold']
    
    def _load_model(self):
        model = SimpleNet(
            input_size=self.config['model']['input_size'],
            hidden_sizes=self.config['model']['hidden_sizes'],
            output_size=self.config['model']['output_size']
        )
        
        model_path = Path(self.config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def preprocess(self, data):
        """Convert input data to tensor"""
        return torch.FloatTensor(data).to(self.device)
    
    def predict(self, input_data):
        """Make prediction on input data"""
        try:
            with torch.no_grad():
                input_tensor = self.preprocess(input_data)
                output = self.model(input_tensor)
                predictions = (output.squeeze() > self.threshold).cpu().numpy()
                probabilities = output.squeeze().cpu().numpy()
                return predictions, probabilities
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize Flask app
app = Flask(__name__)

# Initialize model
inferencer = ModelInference(config)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = np.array(data['features'])
        if features.shape[1] != config['model']['input_size']:
            return jsonify({
                'error': f'Expected {config["model"]["input_size"]} features, got {features.shape[1]}'
            }), 400
        
        predictions, probabilities = inferencer.predict(features)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(
        host=config['server']['host'],
        port=config['server']['port'],
        debug=config['server']['debug']
    )