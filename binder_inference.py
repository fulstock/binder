#!/usr/bin/env python3
"""
Lightweight Binder inference class for loading and using trained models.
This module is designed for inference-only usage in other applications.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer
from src.model import Binder

logger = logging.getLogger(__name__)


class BinderInference:
    """
    Lightweight class for Binder model inference.
    Loads models saved in safetensors format from BinderTraining.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the inference class with a trained model.
        
        Args:
            model_path (str): Path to the saved model directory (with safetensors).
            device (str): Device to run inference on ("auto", "cpu", "cuda", etc.).
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.entity_types = []
        self.entity_type_str_to_id = {}
        self.max_seq_length = 384
        self.config = {}
        
        # Load the model
        self.load_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        device = torch.device(device)
        logger.info(f"🖥️ Using device: {device}")
        return device
    
    def load_model(self):
        """Load the trained model and its components."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        logger.info(f"📂 Loading Binder model from: {self.model_path}")
        
        # Load inference config
        config_path = os.path.join(self.model_path, "inference_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            
            self.entity_types = self.config.get("entity_types", [])
            self.entity_type_str_to_id = {t: i for i, t in enumerate(self.entity_types)}
            self.max_seq_length = self.config.get("max_seq_length", 384)
            
            logger.info(f"📋 Loaded {len(self.entity_types)} entity types")
        else:
            logger.warning("⚠️ No inference_config.json found")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("🔤 Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            raise
        
        # Load model
        try:
            self.model = Binder.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info("🤖 Model loaded successfully from safetensors")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        logger.info("✅ Binder model ready for inference!")
    
    def predict(self, text: str, return_scores: bool = False) -> List[Dict[str, Any]]:
        """
        Perform named entity recognition on input text.
        
        Args:
            text (str): Input text to process.
            return_scores (bool): Whether to return confidence scores.
            
        Returns:
            List[Dict]: List of detected entities with their spans and types.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Post-process results (simplified version)
        # Note: This is a basic implementation. You may need to adapt
        # based on your specific model output format and requirements.
        entities = self._postprocess_outputs(
            outputs, 
            inputs["offset_mapping"][0], 
            text,
            return_scores
        )
        
        return entities
    
    def _postprocess_outputs(self, outputs, offset_mapping, text: str, return_scores: bool) -> List[Dict[str, Any]]:
        """
        Post-process model outputs to extract entities.
        This is a simplified implementation - you may need to adapt based on your model.
        """
        # This is a placeholder implementation
        # You'll need to implement the actual post-processing logic
        # based on how your Binder model outputs are structured
        
        entities = []
        
        # Example structure (adapt to your model's actual output format):
        # - Extract entity spans and types from model outputs
        # - Map back to original text using offset_mapping
        # - Create entity dictionaries
        
        logger.warning("⚠️ Post-processing implementation is simplified. Adapt based on your model's output format.")
        
        return entities
    
    def predict_batch(self, texts: List[str], return_scores: bool = False) -> List[List[Dict[str, Any]]]:
        """
        Perform batch inference on multiple texts.
        
        Args:
            texts (List[str]): List of input texts.
            return_scores (bool): Whether to return confidence scores.
            
        Returns:
            List[List[Dict]]: List of entity lists for each input text.
        """
        results = []
        for text in texts:
            entities = self.predict(text, return_scores)
            results.append(entities)
        return results
    
    def get_entity_types(self) -> List[str]:
        """Get the list of supported entity types."""
        return self.entity_types.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "entity_types": self.entity_types,
            "max_seq_length": self.max_seq_length,
            "num_entity_types": len(self.entity_types),
            "config": self.config,
        }


# Example usage function
def example_usage():
    """Example of how to use BinderInference."""
    print("🧪 BinderInference Example Usage")
    
    # Path to your trained model (saved with safetensors)
    model_path = "./my_custom_binder_model"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train a model first using BinderTraining")
        return
    
    try:
        # Initialize inference
        inference = BinderInference(model_path)
        
        # Get model info
        info = inference.get_model_info()
        print(f"📊 Model supports {info['num_entity_types']} entity types")
        
        # Single prediction
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        entities = inference.predict(text)
        print(f"🔍 Found {len(entities)} entities in: {text}")
        
        # Batch prediction
        texts = [
            "Microsoft was founded by Bill Gates.",
            "Google is headquartered in Mountain View."
        ]
        batch_results = inference.predict_batch(texts)
        print(f"📦 Batch processed {len(texts)} texts")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    example_usage() 