"""
Flask web application for IFS Activity Transformer.
"""
import os
import json
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from app.database import OracleDBConnection
from app.model import ActivityClassifier
import pandas as pd

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Global objects
# NOTE: These global objects are not thread-safe. For production use with
# multiple workers, consider using Redis or a proper job queue system like Celery.
db_connection = None
classifier = None
training_status = {
    'is_training': False,
    'progress': 0,
    'message': 'Not started',
    'metrics': {}
}


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/database')
def database_page():
    """Database connection and data loading page."""
    return render_template('database.html')


@app.route('/api/database/connect', methods=['POST'])
def connect_database():
    """Connect to Oracle database."""
    global db_connection
    
    try:
        db_connection = OracleDBConnection()
        success = db_connection.connect()
        
        if success:
            stats = db_connection.get_activity_statistics()
            return jsonify({
                'success': True,
                'message': 'Connected to database successfully',
                'statistics': stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to connect to database'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/api/database/fetch-data', methods=['POST'])
def fetch_data():
    """Fetch training data from database."""
    global db_connection
    
    if db_connection is None:
        return jsonify({
            'success': False,
            'message': 'Not connected to database'
        }), 400
    
    try:
        data = request.get_json()
        custom_query = data.get('query', None)
        
        df = db_connection.fetch_training_data(custom_query)
        
        if df.empty:
            return jsonify({
                'success': False,
                'message': 'No data retrieved from database'
            }), 404
        
        # Save data to file
        data_path = 'data/training_data.csv'
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        
        return jsonify({
            'success': True,
            'message': f'Retrieved {len(df)} records',
            'preview': df.head(10).to_dict('records'),
            'columns': df.columns.tolist(),
            'row_count': len(df)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/train')
def train_page():
    """Model training page."""
    return render_template('train.html')


@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start model training."""
    global classifier, training_status
    
    try:
        # Get training parameters
        data = request.get_json()
        text_column = data.get('text_column', 'activity_description')
        label_column = data.get('label_column', 'new_activity_path')
        test_size = data.get('test_size', 0.2)
        
        # Check if data exists
        data_path = 'data/training_data.csv'
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'message': 'No training data found. Please fetch data first.'
            }), 400
        
        # Update status
        training_status['is_training'] = True
        training_status['progress'] = 10
        training_status['message'] = 'Loading data...'
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Initialize classifier
        classifier = ActivityClassifier()
        
        training_status['progress'] = 20
        training_status['message'] = 'Preparing data...'
        
        # Prepare data
        train_dataset, val_dataset, num_labels = classifier.prepare_data(
            df, text_column, label_column, test_size
        )
        
        training_status['progress'] = 40
        training_status['message'] = 'Initializing model...'
        
        # Initialize model
        classifier.initialize_model(num_labels)
        
        training_status['progress'] = 50
        training_status['message'] = 'Training model...'
        
        # Train model
        metrics = classifier.train(train_dataset, val_dataset)
        
        training_status['progress'] = 100
        training_status['message'] = 'Training complete!'
        training_status['metrics'] = metrics
        training_status['is_training'] = False
        
        return jsonify({
            'success': True,
            'message': 'Training completed successfully',
            'metrics': metrics
        })
    except Exception as e:
        training_status['is_training'] = False
        training_status['message'] = f'Error: {str(e)}'
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get current training status."""
    return jsonify(training_status)


@app.route('/predict')
def predict_page():
    """Prediction page."""
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new activities."""
    global classifier
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'success': False,
                'message': 'No texts provided'
            }), 400
        
        # Load model if not already loaded
        if classifier is None or classifier.model is None:
            model_dir = os.getenv('SAVE_DIR', './models/saved')
            if not os.path.exists(model_dir):
                return jsonify({
                    'success': False,
                    'message': 'No trained model found. Please train a model first.'
                }), 404
            
            classifier = ActivityClassifier()
            classifier.load_model(model_dir)
        
        # Make predictions
        predictions = classifier.predict(texts)
        
        results = []
        for i, (text, (label, confidence)) in enumerate(zip(texts, predictions)):
            results.append({
                'text': text,
                'predicted_activity': label,
                'confidence': float(confidence)
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Upload CSV file with training data."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'message': 'File must be a CSV'
            }), 400
        
        # Save file
        data_path = 'data/training_data.csv'
        os.makedirs('data', exist_ok=True)
        file.save(data_path)
        
        # Read and validate
        df = pd.read_csv(data_path)
        
        return jsonify({
            'success': True,
            'message': f'Uploaded {len(df)} records',
            'columns': df.columns.tolist(),
            'preview': df.head(10).to_dict('records'),
            'row_count': len(df)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
