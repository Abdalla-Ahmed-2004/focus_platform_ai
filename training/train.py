from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. إعداد المسارات وقراءة المصفوفات الناتجة من الـ Preprocess
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

X_PATH = DATA_DIR / "lstm_X.npy"
Y_PATH = DATA_DIR / "lstm_y.npy"
MODEL_SAVE_PATH = MODELS_DIR / "lstm_knowledge_tracing_model.h5"

def train_deep_knowledge_tracing():
    print("⏳ جاري تحميل مصفوفات البيانات المجهزة...")
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError("❌ ملفات المعالجة .npy غير موجودة، قم بتشغيل preprocess_lstm.py أولاً.")
        
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    # الـ LSTM يحتاج المستهدف (y) بأبعاد ثلاثية أيضاً لتوقع كل خطوة زمنية [Batch, Timesteps, 1]
    y = np.expand_dims(y, axis=-1)
    
    print(f"📊 حجم البيانات المدخلة للتدريب: {X.shape}")
    print(f"📊 حجم البيانات المستهدفة للتدريب: {y.shape}")
    
    # تقسيم البيانات: 80% للتدريب و 20% للتحقق والاختبار (Validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 2. بناء معمارية شبكة الـ LSTM (Deep Knowledge Tracing)
    print("\n🧠 جاري بناء شبكة التعلم العميق (LSTM)...")
    model = Sequential([
        # طبقة الـ Masking: تخبر الموديل أن يتجاهل تماماً أي خطوة زمنية قيمتها -1.0 (الخانات الفارغة)
        Masking(mask_value=-1.0, input_shape=(X.shape[1], X.shape[2])),
        
        # الطبقة الزمنية الأولى للـ LSTM لامتصاص منحنى النسيان والـ Momentum للطالب
        LSTM(64, return_sequences=True),
        Dropout(0.2), # لحماية الموديل من الحفظ الصم (Overfitting)
        
        # الطبقة الزمنية الثانية لتركيز الأنماط وتثبيتها عبر الزمن
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        
        # طبقة إخراج احتمالية لكل خطوة زمنية (توقع الإجابة على السؤال القادم)
        Dense(1, activation='sigmoid')
    ])
    
    # 3. إعداد الـ Compiler مع مقياس الـ AUC الشهير في الأبحاث لقياس الدقة
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy', # لأن الهدف تصنيفي (صح 1 أو خطأ 0)
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # 4. إعداد صمامات الأمان أثناء التدريب (Callbacks)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # حفظ أفضل نسخة للموديل بناءً على دقة الـ Validation AUC
        ModelCheckpoint(filepath=str(MODEL_SAVE_PATH), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        # إيقاف التدريب تلقائياً لو الموديل جاب آخره ومبقاش بيتحسن عشان ميوفرش وقت السيرفر
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1)
    ]
    
    # 5. انطلاق عملية التدريب (Training Epochs)
    print("\n🚀 انطلاق تدريب الموديل العميق الآن...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20, # يمكنك زيادتها لـ 50 لو السيرفر سريع عندك
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print(f"✅ تم تدريب الموديل بنجاح وحفظ النسخة العبقرية في -> {MODEL_SAVE_PATH}")
    print("="*60)

if __name__ == "__main__":
    train_deep_knowledge_tracing()