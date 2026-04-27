
import os
import json
import yaml
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

# Mocking the environment before importing routes
os.environ["MODELS_DIR"] = "test_models"

import sys
sys.path.append(os.getcwd())

# Import the routes
from backend.api import routes

class TestBackendRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("test_models"):
            os.makedirs("test_models/Alpha_V1_Hybrid_CNNLSTM")
        
        # Create a dummy config
        config = {
            "project": {"feature_pipeline": "default"},
            "model": {"modality": "MultiModal"},
            "data": {"processed_dir": "data/processed_market"}
        }
        with open("test_models/Alpha_V1_Hybrid_CNNLSTM/config.yaml", "w") as f:
            yaml.dump(config, f)
            
        # Create a dummy stats with NaN
        stats = {
            "Sharpe Ratio": float('nan'),
            "Win Rate [%]": 65.0
        }
        with open("test_models/Alpha_V1_Hybrid_CNNLSTM/stats_validation.json", "w") as f:
            json.dump(stats, f)

        # Create dummy model.joblib
        with open("test_models/Alpha_V1_Hybrid_CNNLSTM/model.joblib", "w") as f:
            f.write("dummy")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")

    @patch("backend.api.routes.MODELS_DIR", "test_models")
    def test_get_models(self):
        result = routes.get_models()
        self.assertIn("Alpha_V1_Hybrid_CNNLSTM", result["models"])

    @patch("backend.api.routes.MODELS_DIR", "test_models")
    def test_get_model_details_nan_handling(self):
        result = routes.get_model_details("Alpha_V1_Hybrid_CNNLSTM")
        self.assertIsNone(result["stats"]["Sharpe Ratio"])
        self.assertEqual(result["stats"]["Win Rate [%]"], 65.0)

    @patch("backend.api.routes.fetch_live_data")
    @patch("backend.api.routes.MODELS_DIR", "test_models")
    @patch("joblib.load")
    @patch("src.models.model_factory.ModelFactory.get_model")
    @patch("os.path.exists")
    def test_get_live_prediction_market_closed(self, mock_exists, mock_get_model, mock_load, mock_fetch):
        # Mock paths exists
        def side_effect(path):
            if "model.joblib" in path: return True
            if "config.yaml" in path: return True
            return False
        mock_exists.side_effect = side_effect

        # Mock model loading
        mock_load.return_value = {
            "model_type": "lstm",
            "model_params": {},
            "scaler": MagicMock(),
            "feature_cols": ["Close"],
            "atr_multiplier": 3.0
        }
        mock_get_model.return_value = MagicMock()

        # Mock market closed
        mock_fetch.return_value = None
        
        with self.assertRaises(HTTPException) as cm:
            routes.get_live_prediction("Alpha_V1_Hybrid_CNNLSTM")
        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("Market Closed", cm.exception.detail)

    @patch("backend.api.routes.fetch_live_data")
    @patch("backend.api.routes.MODELS_DIR", "test_models")
    @patch("joblib.load")
    @patch("src.models.model_factory.ModelFactory.get_model")
    @patch("os.path.exists")
    def test_get_live_prediction_multi_modal_flag(self, mock_exists, mock_get_model, mock_load, mock_fetch):
        # Mock paths exists
        def side_effect(path):
            if "model.joblib" in path: return True
            if "config.yaml" in path: return True
            if "model_state.joblib" in path: return True
            return False
        mock_exists.side_effect = side_effect

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 0]) # need at least 2 for recent_preds logic
        mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
        mock_get_model.return_value = mock_model
        
        mock_load.return_value = {
            "model_type": "lstm",
            "model_params": {},
            "scaler": MagicMock(),
            "feature_cols": ["Close"],
            "atr_multiplier": 3.0
        }
        
        # Mock data - need enough data for tail(600) and other ops
        # Actually I'll just mock it to have enough rows
        df = pd.DataFrame({
            "time": [pd.Timestamp.now()] * 600,
            "Open": [1.1] * 600, "High": [1.2] * 600, "Low": [1.0] * 600, "Close": [1.15] * 600,
            "ATRr_14": [0.01] * 600
        })
        mock_fetch.return_value = df
        
        # We need to mock scaler.transform to return same shape
        mock_load.return_value["scaler"].transform.return_value = np.zeros((600, 1))
        mock_model.predict.return_value = np.zeros(600)
        mock_model.predict_proba.return_value = np.zeros((600, 3))

        result = routes.get_live_prediction("Alpha_V1_Hybrid_CNNLSTM")
        self.assertTrue(result["is_multi_modal"])

    def test_get_latest_news_empty_fields(self):
        # Create a dummy CSV with empty fields
        test_csv = "test_news.csv"
        df = pd.DataFrame({
            "time": ["2026-03-01 10:00:00", "2026-03-01 11:00:00"],
            "headline": ["Test News 1", ""],
            "sentiment": [0.5, np.nan]
        })
        # Note: the code re-reads the CSV from a hardcoded path in get_latest_news
        # or we can mock pd.read_csv to return our df.
        
        with patch("backend.api.routes.pd.read_csv", return_value=df):
             with patch("backend.api.routes.os.path.exists", return_value=True):
                 result = routes.get_latest_news()
                 news = result["news"]
                 # After sort (ascending=False): 11:00 is first (headline ""), 10:00 is second (headline "Test News 1")
                 self.assertEqual(news[0]["headline"], "")
                 self.assertEqual(news[0]["sentiment"], "") # fillna("")
        
        if os.path.exists(test_csv):
            os.remove(test_csv)

if __name__ == "__main__":
    unittest.main()
