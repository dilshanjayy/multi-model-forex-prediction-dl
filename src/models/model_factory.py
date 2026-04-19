from typing import Any, Dict
from src.models.rf_model import RFModel
from src.models.transformer_model import TransformerModel

class ModelFactory:
    """
    Registry for model implementations.
    New models should be registered in the 'MODELS' dictionary.
    """
    MODELS = {
        "RandomForest": RFModel,
        "Transformer": TransformerModel,
        # Future additions:
        # "LSTM": LSTMModel,
        # "CNN": CNNModel,
    }

    @staticmethod
    def get_model(model_type: str, config: Dict[str, Any]):
        """
        Instantiates a model of the specified type with the provided config.
        """
        model_class = ModelFactory.MODELS.get(model_type)
        if not model_class:
            raise ValueError(f"Model type '{model_type}' is not registered.")
        
        return model_class(config)
