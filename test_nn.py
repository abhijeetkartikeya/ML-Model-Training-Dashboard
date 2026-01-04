"""
Quick test of Neural Network training
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.models.neural_network import NeuralNetworkModel

print("Creating test data...")
# Simple test data
X_train = np.random.randn(100, 2)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.randn(20, 2)
y_test = np.random.randint(0, 2, 20)

print("Initializing model...")
model = NeuralNetworkModel()

print("Training model...")
try:
    result = model.train(X_train, y_train, X_test, y_test)
    print(f"✅ Training completed successfully!")
    print(f"Result: {result}")
    
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"✅ Neural network is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
