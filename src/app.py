import os
import io
import mysql.connector
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from PIL import Image
import base64
import logging
from dotenv import load_dotenv
import json
import uuid
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from google.cloud import storage

# Load .env 
load_dotenv()

# Initialize Flask and Flask-RESTX
app = Flask(__name__)
api = Api(app, version='1.0', title='Skin Disease Prediction API',
          description='An API for predicting skin diseases using machine learning',
          doc='/')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Firebase configuration
firebase_path = os.path.join(os.path.dirname(__file__), 'firebase', 'firebaseService.json')
cred = credentials.Certificate(firebase_path)
firebase_admin.initialize_app(cred)

# Storage configuration
storage_client = storage.Client()
bucket = storage_client.get_bucket('bioface')

# Disease data configuration 
blob = bucket.blob('data/disease_info.json')
disease_info_json = blob.download_as_string()
DISEASE_INFO = json.loads(disease_info_json)

# Load disease information
DISEASE_CLASSES = ['acne', 'clear', 'redness', 'wrinkle']

# MySQL configuration
MYSQL_CONFIG = {
    'unix_socket': os.getenv('MYSQL_HOST'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DB')
}

# Environment variables and setup
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
GCP_STORAGE_BUCKET = os.getenv('GCP_STORAGE_BUCKET')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define namespaces swagger
health_ns = api.namespace('health', description='Health checks')
prediction_ns = api.namespace('prediction', description='Skin Disease Prediction Operations')
history_ns = api.namespace('history', description='Prediction History Management')

# Input and Output Models
image_upload_model = api.model('ImageUpload', {
    'image': fields.Raw(required=True, description='Image file to predict')
})

# Description solution herbal
herbal_solution_model = api.model('HerbalSolution', {
    'name': fields.String(description='Name of herbal solution'),
    'benefit': fields.String(description='Benefit of herbal solution'),
    'usage': fields.String(description='Usage instructions'),
    'imageUrl': fields.String(description='Image URL for herbal solution')
})

# Description skincare product
skincare_product_model = api.model('SkincareProduct', {
    'name': fields.String(description='Name of skincare product'),
    'imageUrl': fields.String(description='Image URL for product')
})

# Prediction response
prediction_result_model = api.model('PredictionResult', {
    'status': fields.String(required=True, description='Status of prediction'),
    'face_disease': fields.String(description='Predicted skin disease'),
    'disease_accuracy': fields.String(description='Accuracy of prediction'),
    'disease_description': fields.String(description='Description of the disease'),
    'image_url': fields.String(description='URL of uploaded image'),
    'prediction_detail': fields.Raw(description='Detailed prediction information'),
    'recomendation': fields.Raw(description='Recommendations for treatment')
})

# Middleware for Firebase token verification
def verify_firebase_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                id_token = auth_header.split('Bearer ')[1]
        
        if not id_token:
            api.abort(401, "No token provided")
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.user = decoded_token
            return f(*args, **kwargs)
        
        except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError):
            api.abort(401, "Invalid or expired token")
    
    return decorated_function

# Image upload parser
image_upload_parser = reqparse.RequestParser()
image_upload_parser.add_argument('image', type=FileStorage, location='files', required=True, 
help='Image file for skin disease prediction')

# Helper Functions
def update_database_schema():
    try:
        mysql_config = {
            'host': os.getenv('MYSQL_HOST', 'mysql'),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'database': os.getenv('MYSQL_DB')
        }

        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()

        cursor.execute("SHOW TABLES LIKE 'prediction_history'")
        if cursor.fetchone() is None:
            cursor.execute('''
                CREATE TABLE prediction_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(255),
                    user_email VARCHAR(255),
                    top_prediction VARCHAR(50),
                    top_prediction_accuracy FLOAT,
                    prediction_details_accuracy TEXT,
                    filename VARCHAR(255),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            logging.info("Database schema created")
        
        cursor.close()
        conn.close()

    except mysql.connector.Error as e:
        logging.error(f"Failed to update database schema: {e}")

def init_db():
    update_database_schema()

# Global variable to store loaded model
MODEL_INSTANCE = None

def load_model():
    global MODEL_INSTANCE

    if MODEL_INSTANCE is not None:
        return MODEL_INSTANCE

    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(GCP_STORAGE_BUCKET)
        blob = bucket.blob('model/model.h5')
        temp_file_path = os.path.join(UPLOAD_FOLDER, 'model.h5')

        if os.path.exists(temp_file_path):
            logging.info("Loading model from local file")
            MODEL_INSTANCE = tf.keras.models.load_model(temp_file_path)
        else:
            logging.info("Downloading model from GCP Storage")
            blob.download_to_filename(temp_file_path)
            MODEL_INSTANCE = tf.keras.models.load_model(temp_file_path)

        logging.info(f"Model loaded from GCP Storage bucket: {GCP_STORAGE_BUCKET}")
        return MODEL_INSTANCE

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def preprocess_image(image_file, target_size=(224, 224)):
    try:
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize(target_size)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        logging.error(f"Failed to process image: {e}")
        return None

def save_prediction_history(filename, top_prediction, predictions, top_prediction_accuracy, user_id, user_email):
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO prediction_history 
            (filename, top_prediction, prediction_details_accuracy, top_prediction_accuracy, user_id, user_email) 
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (
            filename, 
            top_prediction, 
            str(predictions), 
            top_prediction_accuracy,
            user_id,
            user_email
        ))

        conn.commit()
        logging.info(f"Prediction history saved for {filename}")
    except mysql.connector.Error as e:
        logging.error(f"Failed to save prediction history: {e}")
    finally:
        cursor.close()
        conn.close()

# Security Definition
api.authorizations = {
    'apiKey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization'
    }
}

# Health Check Route
@health_ns.route('')
class HealthCheck(Resource):
    def get(self):
        """
        Health check endpoint
        """
        return {
            "status": "healthy",
            "message": "API is up and running",
            "components": {
                "database": self.check_database(),
                "model": self.check_model(),
                "storage": self.check_storage()
            }
        }, 200

    def check_database(self):
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            conn.ping(reconnect=True)
            conn.close()
            return "connected"
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            return "disconnected"

    def check_model(self):
        try:
            model = load_model()
            return "loaded" if model is not None else "not loaded"
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            return "error"

    def check_storage(self):
        try:
            storage_client = storage.Client()
            storage_client.get_bucket(GCP_STORAGE_BUCKET)
            return "accessible"
        except Exception as e:
            logging.error(f"Storage access error: {e}")
            return "inaccessible"

# POST - Prediction
@prediction_ns.route('')
class SkinDiseasePrediction(Resource):
    @prediction_ns.doc('predict_skin_disease', 
                        security='apiKey',
                        responses={
                            200: 'Successful prediction',
                            400: 'Invalid image',
                            401: 'Unauthorized',
                            500: 'Model prediction error'
                        })
    @prediction_ns.expect(image_upload_parser)
    @verify_firebase_token
    def post(self):
        """
        Predict skin disease from an uploaded image
        Requires Firebase authentication token
        """
        args = image_upload_parser.parse_args()
        image_file = args['image']
        
        if image_file.filename == '':
            api.abort(400, "No file selected")
        
        try:
            filename = f"{uuid.uuid4().hex}.{image_file.filename.split('.')[-1]}"
            
            # Save image to Cloud Storage
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(GCP_STORAGE_BUCKET)
            blob = bucket.blob(f'user_image/{filename}')
            image_file.seek(0)  # Reset file pointer
            blob.upload_from_file(image_file)
            
            model = load_model()
            if model is None:
                api.abort(500, "Failed to load model")
            
            processed_image = preprocess_image(image_file)
            if processed_image is None:
                api.abort(400, "Failed to process image")
            
            predictions = model.predict(processed_image)[0]
            results = {
                DISEASE_CLASSES[i]: f"{round(float(predictions[i]) * 100, 1)}%" 
                for i in range(len(DISEASE_CLASSES))
            }
            
            top_prediction = max(results, key=lambda k: float(results[k].rstrip('%')))
            top_prediction_accuracy = f"{round(float(results[top_prediction].rstrip('%')),1)}%"

            save_prediction_history(
                filename, 
                top_prediction, 
                results, 
                float(top_prediction_accuracy.rstrip('%')),
                request.user['uid'],
                request.user.get('email', '')
            )
            
            disease_info = DISEASE_INFO.get(top_prediction, {})
            
            response = {
                "status": "Prediction Success",
                "face_disease": top_prediction.capitalize(),
                "disease_accuracy": top_prediction_accuracy,
                "disease_description": disease_info.get("description", ""),
                "image_url": f"https://storage.googleapis.com/{GCP_STORAGE_BUCKET}/{filename}",
                "prediction_detail": {
                    "causes": disease_info.get("causes", []),
                    "detail_disease_accuracy": {
                        k: f"{round(float(v.rstrip('%')),1)}%"
                        for k, v in results.items()
                    }
                },
                "recomendation": {
                    "herbalSolutions": [
                        {
                            "name": solution["name"],
                            "benefit": solution["benefit"],
                            "usage": solution["usage"],
                            "imageUrl": solution.get("imageUrl", "")
                        }
                        for solution in disease_info.get("herbalSolutions", [])
                    ],
                    "skincareProducts": [
                        {
                            "name": product["name"],
                            "imageUrl": product.get("imageUrl", "")
                        }
                        for product in disease_info.get("skincareProducts", [])
                    ]
                }
            }
            
            return response, 200
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            api.abort(500, "Error occurred during prediction")

# GET - History User
@history_ns.route('')
class PredictionHistory(Resource):
    @history_ns.doc('get_prediction_history', 
                    security='apiKey',
                    responses={
                        200: 'Successful retrieval of prediction history',
                        401: 'Unauthorized',
                        404: 'No prediction history found'
                    })
    @verify_firebase_token
    def get(self):
        """
        Retrieve user's prediction history
        Requires Firebase authentication token
        """
        try:
            user_id = request.user['uid']
            user_email = request.user.get('email', '')

            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor(dictionary=True)

            cursor.execute('''
                SELECT * FROM prediction_history 
                WHERE user_id = %s OR user_email = %s
                ORDER BY timestamp DESC LIMIT 10
            ''', (user_id, user_email))
            
            predictions = cursor.fetchall()

            if not predictions:
                api.abort(404, "No prediction history found")

            formatted_predictions = []
            
            for prediction in predictions:
                timestamp = prediction['timestamp'].strftime("%a, %d %b %Y %H:%M:%S GMT")
                disease_accuracy = f"{round(prediction['top_prediction_accuracy'], 1)}%"
                face_disease = prediction['top_prediction'].capitalize()

                disease_info = DISEASE_INFO.get(face_disease.lower(), {})

                disease_description = disease_info.get('description', 'Deskripsi tidak tersedia.')
                causes = disease_info.get('causes', [])
                herbal_solutions = disease_info.get('herbalSolutions', [])
                skincare_products = disease_info.get('skincareProducts', [])

                formatted_predictions.append({
                    "disease_accuracy": disease_accuracy,
                    "disease_description": disease_description,
                    "face_disease": face_disease,
                    "id": prediction['id'],
                    "image_url": f"https://storage.googleapis.com/{GCP_STORAGE_BUCKET}/{prediction['filename']}",
                    "prediction_detail": {
                        "causes": causes,
                        "detail_disease_accuracy": prediction['prediction_details_accuracy'],
                    },
                    "recomendation": {
                        "herbalSolutions": herbal_solutions,
                        "skincareProducts": skincare_products,
                    },
                    "timestamp": timestamp
                })

            response = {
                "status": "Success",
                "email": prediction['user_email'],
                "predictions": formatted_predictions
            }

            return response, 200

        except mysql.connector.Error as e:
            logging.error(f"Failed to fetch prediction history: {e}")
            api.abort(500, "Failed to fetch prediction history")

        finally:
            cursor.close()
            conn.close()

# GET - History User By ID
# DELETE - History User By ID
@history_ns.route('/<int:id>')
class PredictionHistoryByID(Resource):
    @history_ns.doc('get_prediction_by_id', 
                    security='apiKey',
                    responses={
                        200: 'Successful retrieval of specific prediction',
                        401: 'Unauthorized',
                        404: 'Prediction not found'
                    })
    @verify_firebase_token
    def get(self, id):
        """
        Retrieve a specific prediction by its ID
        Requires Firebase authentication token
        """
        try:
            user_id = request.user['uid']
            user_email = request.user.get('email', '')

            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor(dictionary=True)

            cursor.execute('''
                SELECT * FROM prediction_history
                WHERE id = %s AND (user_id = %s OR user_email = %s)
            ''', (id, user_id, user_email))

            prediction = cursor.fetchone()

            if not prediction:
                api.abort(404, f"No prediction found for ID {id}")

            timestamp = prediction['timestamp'].strftime("%a, %d %b %Y %H:%M:%S GMT")
            disease_accuracy = f"{round(prediction['top_prediction_accuracy'], 1)}%"
            face_disease = prediction['top_prediction'].capitalize()

            disease_info = DISEASE_INFO.get(face_disease.lower(), {})

            disease_description = disease_info.get('description', 'Deskripsi tidak tersedia.')
            causes = disease_info.get('causes', [])
            herbal_solutions = disease_info.get('herbalSolutions', [])
            skincare_products = disease_info.get('skincareProducts', [])

            response = {
                "status": "success",
                "email": prediction['user_email'],
                "prediction": {
                    "id": prediction['id'],
                    "filename": prediction['filename'],
                    "disease_accuracy": disease_accuracy,
                    "disease_description": disease_description,
                    "face_disease": face_disease,
                    "image_url": f"https://storage.googleapis.com/{GCP_STORAGE_BUCKET}/{prediction['filename']}",
                    "prediction_detail": {
                        "causes": causes,
                        "detail_disease_accuracy": prediction['prediction_details_accuracy'],
                    },
                    "timestamp": timestamp,
                    "recomendation": {
                        "herbalSolutions": herbal_solutions,
                        "skincareProducts": skincare_products,
                    }
                }
            }

            return response, 200

        except mysql.connector.Error as e:
            logging.error(f"Failed to fetch prediction by ID: {e}")
            api.abort(500, "Failed to fetch prediction by ID")

        finally:
            cursor.close()
            conn.close()

    @history_ns.doc('delete_prediction_by_id', 
                    security='apiKey',
                    responses={
                        200: 'Successful deletion of prediction',
                        401: 'Unauthorized',
                        404: 'Prediction not found'
                    })
    @verify_firebase_token
    def delete(self, id):
        """
        Delete a specific prediction by its ID
        Requires Firebase authentication token
        """
        try:
            user_id = request.user['uid']
            user_email = request.user.get('email', '')
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor(dictionary=True)

            cursor.execute('''
                SELECT id FROM prediction_history
                WHERE id = %s AND (user_id = %s OR user_email = %s)
            ''', (id, user_id, user_email))
            prediction = cursor.fetchone()
            
            if not prediction:
                api.abort(404, f"No prediction found for ID {id} or unauthorized access")
        
            cursor.execute('''
                DELETE FROM prediction_history
                WHERE id = %s
            ''', (id,))
            conn.commit()
            
            return {
                "status": "Success", 
                "message": f"Prediction history entry with ID {id} deleted successfully"
            }, 200
        
        except mysql.connector.Error as e:
            logging.error(f"Failed to delete prediction history entry: {e}")
            api.abort(500, "Failed to delete prediction history entry")
        
        finally:
            cursor.close()
            conn.close()

# DELETE - All History User
@history_ns.route('/delete-all')
class DeleteAllPredictionHistory(Resource):
    @history_ns.doc('delete_all_history', 
                    security='apiKey',
                    responses={
                        200: 'Successfully deletecd all prediction history',
                        401: 'Unauthorized',
                        500: 'Error during deletion'
                    })
    @verify_firebase_token
    def delete(self):
        """
        Delete all prediction history for the authenticated user
        Requires Firebase authentication token
        """
        try:
            user_id = request.user['uid']
            user_email = request.user.get('email', '')
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute('''
                DELETE FROM prediction_history
                WHERE user_id = %s OR user_email = %s
            ''', (user_id, user_email))
            
            rows_deleted = cursor.rowcount
            conn.commit()
            
            return {
                "status": "Success", 
                "message": f"{rows_deleted} prediction history entries deleted successfully"
            }, 200
        
        except mysql.connector.Error as e:
            logging.error(f"Failed to delete all prediction history entries: {e}")
            api.abort(500, "Failed to delete prediction history entries")
        
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=8080, debug=True) 