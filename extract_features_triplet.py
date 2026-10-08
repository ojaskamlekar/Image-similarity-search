"""
Feature Extraction using Trained Triplet Network
Extracts embeddings for all images in the dataset
"""

import os
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from tensorflow import keras
from triplet_model import triplet_loss


class TripletFeatureExtractor:
    """Extract embeddings using trained triplet network"""
    
    def __init__(self, model_path='triplet_base_final.h5'):
        """
        Load the trained base network
        
        Args:
            model_path: Path to the saved trained model
        """
        print("=" * 60)
        print("📦 Loading Trained Triplet Network")
        print("=" * 60)
        
        try:
            self.model = keras.models.load_model(
                model_path,
                custom_objects={'triplet_loss': triplet_loss},
                compile=False
            )
            print(f"✅ Model loaded from: {model_path}")
            print(f"📊 Model parameters: {self.model.count_params():,}")
            print(f"📐 Input shape: {self.model.input_shape}")
            print(f"📐 Output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Make sure you have:")
            print("   1. Trained the model (run train_triplet.py)")
            print("   2. The model file exists: triplet_base_final.h5")
            exit(1)
        
        print("=" * 60)
    
    def preprocess_image(self, image_path, target_size=None):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing
        
        Returns:
            Preprocessed image array ready for model input
        """
        if target_size is None:
            target_size = (self.model.input_shape[2], self.model.input_shape[1])
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def extract_features(self, image_path):
        """
        Extract embedding for a single image
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Embedding vector (flattened numpy array)
        """
        img = self.preprocess_image(image_path)
        embedding = self.model(img, training=False).numpy()
        return embedding.flatten()

    def extract_paths(self, image_paths, batch_size=16):
        """Batch inference for evaluation/indexing; fail on unreadable images."""
        if batch_size < 1 or not image_paths:
            raise ValueError('Need images and a positive batch size')
        outputs = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            inputs = np.concatenate([self.preprocess_image(str(p)) for p in batch_paths])
            outputs.append(self.model(inputs, training=False).numpy())
            if start % (batch_size * 20) == 0 or start + batch_size >= len(image_paths):
                print(f'Embedded {min(start + batch_size, len(image_paths))}/{len(image_paths)} images', flush=True)
        return np.concatenate(outputs)
    
    def extract_all_features(self, image_folder, output_prefix='triplet'):
        """
        Extract embeddings for all images in folder
        
        Args:
            image_folder: Folder containing images
            output_prefix: Prefix for output files
        
        Returns:
            features_list: List of feature vectors
            image_names: List of corresponding image filenames
        """
        print("\n" + "=" * 60)
        print("🔍 Extracting Features from Dataset")
        print("=" * 60)
        print(f"📁 Image folder: {image_folder}")
        
        features_list = []
        image_names = []
        failed_images = []
        
        # Get all image files
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        total_images = len(image_files)
        print(f"📊 Total images found: {total_images}")
        print("-" * 60)
        
        for i, filename in enumerate(image_files, 1):
            img_path = os.path.join(image_folder, filename)
            
            try:
                # Extract features
                features = self.extract_features(img_path)
                features_list.append(features)
                image_names.append(filename)
                
                # Progress indicator
                if i % 10 == 0 or i == total_images:
                    print(f"✓ Processed {i}/{total_images} images ({i/total_images*100:.1f}%)")
                    
            except Exception as e:
                failed_images.append((filename, str(e)))
                print(f"⚠ Failed to process {filename}: {e}")
        
        print("-" * 60)
        print(f"✅ Successfully processed: {len(features_list)}/{total_images} images")
        
        if failed_images:
            print(f"⚠ Failed images: {len(failed_images)}")
            for fname, error in failed_images[:5]:  # Show first 5 failures
                print(f"   - {fname}: {error}")
        
        # Save features to disk
        if len(features_list) > 0:
            features_array = np.array(features_list)
            images_array = np.array(image_names)
            
            features_file = f"{output_prefix}_features.npy"
            images_file = f"{output_prefix}_images.npy"
            
            np.save(features_file, features_array)
            np.save(images_file, images_array)
            
            print("\n" + "=" * 60)
            print("💾 Files Saved")
            print("=" * 60)
            print(f"✅ {features_file}")
            print(f"   Shape: {features_array.shape}")
            print(f"   Size: {features_array.nbytes / 1024:.2f} KB")
            print(f"✅ {images_file}")
            print(f"   Count: {len(images_array)} filenames")
            print("=" * 60)
        else:
            print("❌ No features extracted!")
        
        return features_list, image_names
    
    def visualize_embedding_space(self, features, labels=None, output_file='embedding_space.png'):
        """
        Visualize the embedding space using t-SNE (optional)
        Requires: pip install scikit-learn matplotlib
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            print("\n📊 Generating embedding space visualization...")
            
            # Reduce to 2D using t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            embeddings_2d = tsne.fit_transform(features)
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                       alpha=0.6, s=50, c=range(len(embeddings_2d)), cmap='viridis')
            plt.colorbar(label='Image Index')
            plt.title('Triplet Network Embedding Space (t-SNE)', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"✅ Visualization saved: {output_file}")
            
        except ImportError:
            print("⚠ Skipping visualization (scikit-learn not installed)")
        except Exception as e:
            print(f"⚠ Could not create visualization: {e}")


def main():
    """Main feature extraction function"""
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='triplet_base_final.h5')
    parser.add_argument('--image-folder', default='static/dataset')
    parser.add_argument('--output-prefix', default='triplet')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--no-visualization', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    MODEL_PATH = args.model
    IMAGE_FOLDER = args.image_folder
    OUTPUT_PREFIX = args.output_prefix
    
    print("\n" + "=" * 60)
    print("🎯 TRIPLET FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load trained model
    extractor = TripletFeatureExtractor(MODEL_PATH)
    
    # Step 2: Extract features for all images
    paths = sorted(p for p in Path(IMAGE_FOLDER).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    features = extractor.extract_paths(paths, args.batch_size)
    image_names = [p.name for p in paths]
    Path(OUTPUT_PREFIX).parent.mkdir(parents=True, exist_ok=True)
    np.save(f'{OUTPUT_PREFIX}_features.npy', features)
    np.save(f'{OUTPUT_PREFIX}_images.npy', np.asarray(image_names))
    metadata = {'model_sha256': hashlib.sha256(Path(MODEL_PATH).read_bytes()).hexdigest(),
                'image_count': len(paths), 'embedding_dimension': int(features.shape[1]),
                'preprocessing': {'input_shape': list(extractor.model.input_shape[1:]),
                                  'color': 'RGB', 'dtype': 'float32', 'scale': 'divide by 255'},
                'images': [{'filename': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in paths]}
    Path(f'{OUTPUT_PREFIX}_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    
    # Step 3: Optional visualization
    if len(features) > 1 and not args.no_visualization:
        print("\n📊 Creating embedding space visualization...")
        extractor.visualize_embedding_space(
            np.array(features),
            output_file=f'{OUTPUT_PREFIX}_embedding_space.png'
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"📁 Generated Files:")
    print(f"   ✓ {OUTPUT_PREFIX}_features.npy ({len(features)} embeddings)")
    print(f"   ✓ {OUTPUT_PREFIX}_images.npy ({len(image_names)} filenames)")
    print(f"\n🎯 Next Step: Run Flask app with updated features")
    print(f"   Set IMAGE_SIMILARITY_FEATURE_PREFIX to '{OUTPUT_PREFIX}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
