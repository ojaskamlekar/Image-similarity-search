"""
Triplet Network Model Architecture
Implements Siamese network with triplet loss
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


@keras.utils.register_keras_serializable(package='ImageSimilarity')
class L2Normalization(layers.Layer):
    """Serializable unit-length embedding output, without an anonymous Lambda."""

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)


def triplet_loss(y_true, y_pred, margin=0.2):
    """
    Triplet Loss Function
    
    Goal: Make distance(anchor, positive) + margin < distance(anchor, negative)
    
    Formula: L = max(0, d(A,P) - d(A,N) + margin)
    
    Args:
        y_true: Dummy labels (not used, required by Keras)
        y_pred: Concatenated embeddings [anchor, positive, negative]
        margin: Minimum distance gap between positive and negative pairs
    
    Returns:
        Average triplet loss for the batch
    """
    
    # Get embedding dimension
    embedding_dim = tf.shape(y_pred)[-1] // 3
    
    # Split the concatenated embeddings
    anchor = y_pred[:, 0:embedding_dim]
    positive = y_pred[:, embedding_dim:embedding_dim*2]
    negative = y_pred[:, embedding_dim*2:embedding_dim*3]
    
    # Calculate squared Euclidean distances
    positive_distance = tf.reduce_sum(
        tf.square(anchor - positive), axis=1
    )
    negative_distance = tf.reduce_sum(
        tf.square(anchor - negative), axis=1
    )
    
    # Triplet loss with margin
    loss = tf.maximum(
        positive_distance - negative_distance + margin,
        0.0
    )
    
    return tf.reduce_mean(loss)


def create_base_network(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Create the base CNN network (shared across all three inputs)
    This is the embedding network that learns to map images to feature space
    
    Args:
        input_shape: Input image shape (height, width, channels)
        embedding_dim: Dimension of output embedding vector
    
    Returns:
        Keras Model that outputs normalized embeddings
    """
    
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Convolutional Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    # Convolutional Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    # Convolutional Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.25, name='dropout3')(x)
    
    # Convolutional Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Dense layers for embedding
    x = layers.Dense(512, activation='relu', name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Dropout(0.5, name='dropout_dense1')(x)
    
    x = layers.Dense(256, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='bn_dense2')(x)
    x = layers.Dropout(0.5, name='dropout_dense2')(x)
    
    # Final embedding layer (no activation)
    embeddings = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)
    
    # L2 normalization - makes embeddings unit length
    embeddings = L2Normalization(name='l2_normalization')(embeddings)
    
    model = keras.Model(inputs=inputs, outputs=embeddings, name='base_network')
    return model


def create_triplet_network(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Create the full Triplet Network with three branches
    Uses shared weights across anchor, positive, and negative branches
    
    Args:
        input_shape: Input image shape
        embedding_dim: Dimension of embedding vectors
    
    Returns:
        triplet_model: Full model for training
        base_network: Base network for feature extraction
    """
    
    print("=" * 60)
    print("🏗️  Building Triplet Network Architecture")
    print("=" * 60)
    
    # Create shared base network
    base_network = create_base_network(input_shape, embedding_dim)
    
    print(f"✅ Base network created")
    print(f"   Input shape: {input_shape}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Total parameters: {base_network.count_params():,}")
    
    # Define three inputs (anchor, positive, negative)
    anchor_input = layers.Input(shape=input_shape, name='anchor_input')
    positive_input = layers.Input(shape=input_shape, name='positive_input')
    negative_input = layers.Input(shape=input_shape, name='negative_input')
    
    # Pass all three through the SAME network (shared weights)
    anchor_embedding = base_network(anchor_input)
    positive_embedding = base_network(positive_input)
    negative_embedding = base_network(negative_input)
    
    # Concatenate embeddings for loss calculation
    outputs = layers.Concatenate(name='concatenated_embeddings')([
        anchor_embedding,
        positive_embedding,
        negative_embedding
    ])
    
    # Create the full triplet model
    triplet_model = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=outputs,
        name='triplet_network'
    )
    
    print(f"✅ Triplet network created")
    print(f"   Total trainable parameters: {triplet_model.count_params():,}")
    print("=" * 60)
    
    return triplet_model, base_network


def compile_triplet_model(triplet_model, learning_rate=0.0001, margin=0.2):
    """
    Compile the triplet model with optimizer and loss
    
    Args:
        triplet_model: The triplet network model
        learning_rate: Learning rate for Adam optimizer
        margin: Margin for triplet loss
    """
    
    triplet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y_true, y_pred: triplet_loss(y_true, y_pred, margin=margin)
    )
    
    print(f"✅ Model compiled with:")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: Triplet Loss (margin={margin})")
    
    return triplet_model
