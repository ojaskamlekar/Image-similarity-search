"""Runtime checks use smaller images to keep CPU validation inexpensive."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ('tensorflow', 'cv2', 'flask', 'sklearn'))


@unittest.skipUnless(HAS_RUNTIME, 'Install requirements.txt for TensorFlow/API checks')
class ModelAndApiTests(unittest.TestCase):
    def test_architecture_training_loss_and_serialization(self):
        import tensorflow as tf
        from triplet_model import create_triplet_network, compile_triplet_model, triplet_loss
        model, base = create_triplet_network(input_shape=(32, 32, 3), embedding_dim=8)
        compile_triplet_model(model)
        inputs = np.random.default_rng(42).random((2, 32, 32, 3), dtype=np.float32)
        embeddings = base(inputs, training=False).numpy()
        self.assertEqual(embeddings.shape, (2, 8))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
        self.assertEqual(model.output_shape, (None, 24))
        pred = tf.constant([[1., 0., 1., 0., 0., 1.]])
        self.assertAlmostEqual(float(triplet_loss(None, pred)), 0)
        pred = tf.constant([[1., 0., 0., 1., 1., 0.]])
        self.assertAlmostEqual(float(triplet_loss(None, pred)), 2.2, places=5)
        loss = model.train_on_batch([inputs, inputs, inputs[::-1]], np.zeros((2, 1)))
        self.assertTrue(np.isfinite(loss))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'model.h5')
            base.save(path)
            reloaded = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(base(inputs, training=False).numpy(),
                                       reloaded(inputs, training=False).numpy(), atol=1e-5)
            import cv2
            from extract_features_triplet import TripletFeatureExtractor
            paths = [Path(folder) / f'{i}.jpg' for i in range(3)]
            for i, image_path in enumerate(paths):
                cv2.imwrite(str(image_path), np.full((48, 48, 3), 40 + i * 30, dtype=np.uint8))
            extractor = TripletFeatureExtractor(path)
            batched = extractor.extract_paths(paths, batch_size=2)
            singles = np.stack([extractor.extract_features(str(p)) for p in paths])
            self.assertEqual(batched.shape, (3, 8))
            np.testing.assert_allclose(batched, singles, atol=1e-5)

    def test_api_scores_and_threshold_validation(self):
        import app_triplet as app_module
        with tempfile.TemporaryDirectory() as folder, patch.object(app_module, 'model', object()), \
             patch.object(app_module, 'features', np.array([[1., 0.], [0., 1.]])), \
             patch.object(app_module, 'image_names', np.array(['a.jpg', 'b.jpg'])), \
             patch.object(app_module, 'extract_features_from_image', return_value=np.array([1., 0.])):
            with patch.dict(app_module.app.config, {'UPLOAD_FOLDER': folder, 'TESTING': True}):
                client = app_module.app.test_client()
                response = client.post('/search', data={'image': (io.BytesIO(b'image'), 'query.jpg'),
                                                        'threshold': '0'})
                self.assertEqual(response.status_code, 200)
                body = response.get_json()
                self.assertFalse(body['calibrated_probability'])
                self.assertEqual(body['results'][0]['score'], 1)
                self.assertEqual(body['results'][0]['similarity'], 1)
                for threshold in ('nan', '-0.1', '1.1', 'invalid'):
                    response = client.post('/search', data={'image': (io.BytesIO(b'image'), 'query.jpg'),
                                                            'threshold': threshold})
                    self.assertEqual(response.status_code, 400)
                self.assertFalse(client.get('/stats').get_json()['calibrated_probability'])


if __name__ == '__main__':
    unittest.main()
