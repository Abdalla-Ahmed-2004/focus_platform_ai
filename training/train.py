from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Set paths and read preprocessed arrays from Preprocess
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

X_PATH = DATA_DIR / "lstm_X.npy"
Y_PATH = DATA_DIR / "lstm_y.npy"
MODEL_SAVE_PATH = MODELS_DIR / "lstm_knowledge_tracing_model.h5"

def train_deep_knowledge_tracing():
    print("⏳ Loading preprocessed numpy arrays...")
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError("❌ Preprocessed .npy files not found. Run the preprocessing step first.")
        
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    # LSTM requires the target (y) to be 3D for per-timestep prediction [Batch, Timesteps, 1]
    y = np.expand_dims(y, axis=-1)
    
    print(f"📊 Training input shape: {X.shape}")
    print(f"📊 Training target shape: {y.shape}")
    
    # Split data: 80% for training and 20% for validation/testing
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 2. Build LSTM architecture (Deep Knowledge Tracing)
    print("\n🧠 Building LSTM deep learning model...")
    model = Sequential([
        # Masking layer: instruct the model to ignore any timestep with value -1.0 (padded slots)
        Masking(mask_value=-1.0, input_shape=(X.shape[1], X.shape[2])),
        
        # first LSTM layer to capture forgetting curves and student momentum
        LSTM(64, return_sequences=True),
        Dropout(0.2), # protect the model from overfitting
        
        # second LSTM layer to consolidate temporal patterns
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        
        # output probability per timestep (predict next answer)
        Dense(1, activation='sigmoid')
    ])
    
    # 3. Compile the model and use AUC metric commonly used in research
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy', # because the target is binary (1 correct, 0 incorrect)
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # 4. Set up safeguards during training (callbacks)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # save the best model based on validation AUC
        ModelCheckpoint(filepath=str(MODEL_SAVE_PATH), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        # early stop training if validation AUC stops improving to save time
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1)
    ]
    
    # 5. Start the training process (epochs)
    print("\n🚀 Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20, # you can increase to 50 if your server is fast
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print(f"✅ Model trained successfully and saved to -> {MODEL_SAVE_PATH}")
    print("="*60)

if __name__ == "__main__":
    train_deep_knowledge_tracing()