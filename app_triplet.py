"""
Flask App using Triplet Network Embeddings
Fashion Similarity Search with trained Triplet Network
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow import keras
from werkzeug.utils import secure_filename
from triplet_model import triplet_loss
from similarity_scoring import distance_scores, SCORE_DESCRIPTION

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/dataset'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)

# Load trained triplet network model
print("=" * 60)
print("🚀 Loading Triplet Network Model")
print("=" * 60)

try:
    model = keras.models.load_model(
        'triplet_base_final.h5',
        custom_objects={'triplet_loss': triplet_loss},
        compile=False
    )
    print("✅ Triplet network loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠ Please train the model first using train_triplet.py")
    model = None

# Load pre-computed embeddings
try:
    features = np.load("triplet_features.npy")
    image_names = np.load("triplet_images.npy")
    print(f"✅ Loaded {len(image_names)} image embeddings")
    print(f"📊 Embedding dimension: {features.shape[1]}")
except Exception as e:
    print(f"❌ Error loading features: {e}")
    print("⚠ Please extract features first using extract_features_triplet.py")
    features = np.array([])
    image_names = np.array([])

print("=" * 60)


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def extract_features_from_image(image_path):
    """Extract triplet network embedding from image"""
    if model is None:
        raise ValueError("Model not loaded")
    
    img = preprocess_image(image_path)
    embedding = model.predict(img, verbose=0)
    return embedding.flatten()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """Handle image upload and similarity search using Triplet Network"""
    
    # Validation checks
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the triplet network first.'}), 400
    
    if len(features) == 0:
        return jsonify({'error': 'No embeddings found. Run extract_features_triplet.py first'}), 400
    
    try:
        # Get search parameters
        try:
            num_results = int(request.form.get('num_results', 10))
            similarity_threshold = float(request.form.get('threshold', 0.3))
        except ValueError:
            return jsonify({'error': 'num_results and threshold must be numeric'}), 400
        num_results = min(max(num_results, 5), min(50, len(features)))
        
        if not np.isfinite(similarity_threshold) or not 0 <= similarity_threshold <= 1:
            return jsonify({'error': 'Score threshold must be between 0 and 1'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract embedding using trained triplet network
        query_embedding = extract_features_from_image(filepath)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarity using Euclidean distance (better for normalized embeddings)
        # Lower distance = more similar
        distances = euclidean_distances(query_embedding, features)[0]
        
        # Convert distances to similarity scores (0 to 1, where 1 is most similar)
        # For normalized embeddings, distance is in range [0, 2]
        similarity_scores = distance_scores(distances)
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarity_scores.argsort()[::-1]
        
        # Filter by threshold and limit results
        results = []
        for idx in sorted_indices:
            if len(results) >= num_results:
                break
            if similarity_scores[idx] >= similarity_threshold:
                results.append({
                    'image': f'/static/dataset/{image_names[idx]}',
                    'name': str(image_names[idx]),
                    'score': float(round(similarity_scores[idx], 3)),
                    'similarity': float(round(similarity_scores[idx], 3)),  # Compatibility alias
                    'distance': float(round(distances[idx], 3))
                })
        
        return jsonify({
            'success': True,
            'query_image': f'/static/uploads/{filename}',
            'results': results,
            'total_matches': len(results),
            'method': 'Triplet Network with Euclidean Distance',
            'score_type': 'distance_based_similarity',
            'score_description': SCORE_DESCRIPTION,
            'calibrated_probability': False
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/gallery')
def gallery():
    """Get all images in dataset"""
    try:
        return jsonify({
            'images': [f'/static/dataset/{name}' for name in image_names],
            'total': len(image_names)
        })
    except:
        return jsonify({'images': [], 'total': 0})


@app.route('/stats')
def stats():
    """Get system statistics"""
    return jsonify({
        'model_loaded': model is not None,
        'embeddings_loaded': len(features) > 0,
        'total_images': len(image_names),
        'embedding_dimension': features.shape[1] if len(features) > 0 else 0,
        'model_type': 'Triplet Network',
        'similarity_metric': 'Euclidean Distance',
        'score_description': SCORE_DESCRIPTION,
        'calibrated_probability': False
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Triplet Network Similarity Search Server")
    print("=" * 60)
    print("🎯 Model: Trained Triplet Network")
    print("📊 Similarity Metric: Euclidean Distance in Embedding Space")
    print("=" * 60)
    
    if model is None:
        print("⚠ WARNING: Model not loaded!")
        print("   Run: python train_triplet.py")
    
    if len(features) == 0:
        print("⚠ WARNING: No embeddings loaded!")
        print("   Run: python extract_features_triplet.py")
    
    if model is not None and len(features) > 0:
        print(f"✅ System Ready!")
        print(f"   Images in database: {len(image_names)}")
        print(f"   Embedding dimension: {features.shape[1]}")
    
    print("\n🌐 Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(debug=True, port=5000)
