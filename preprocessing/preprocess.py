from pathlib import Path
import pandas as pd
import numpy as np
import os
import joblib

# 1. إعداد مسارات المشروع التلقائية (عشان الكود يشتغل على أي جهاز بدون تعديل مسارات)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_FILE = DATA_DIR / "2015_100_skill_builders_main_problems.csv"

# ملفات المخرجات اللي البايثون هيعملها بنفسه وهيحفظها بصيغة مصفوفات Numpy سريعة جداً
X_OUTPUT_PATH = DATA_DIR / "lstm_X.npy"
Y_OUTPUT_PATH = DATA_DIR / "lstm_y.npy"
METADATA_PATH = DATA_DIR / "lstm_metadata.pkl"

def load_assistments_file(file_path):
    """دالة ذكية لقراءة ملف الداتا بترميزات مختلفة عشان ميعملش معاك Error أثناء القراءة"""
    file_path = str(file_path)
    for encoding in ['utf-8', 'ISO-8859-1', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    raise ValueError(f"❌ مش قادر أقرأ الملف، تأكد من وجوده في المسار: {file_path}")

def preprocess_for_lstm(file_path, max_seq_len=50):
    print(f"🚀 بدء معالجة البيانات لنموذج الـ LSTM... (طول السلسلة الثابت: {max_seq_len})")
    
    # خطوة أ: قراءة ملف الداتا الأصلي
    df = load_assistments_file(file_path)
    
    # خطوة ب: تنظيف وتوحيد أسماء الأعمدة عشان تطابق الكود
    df = df.rename(columns={'sequence_id': 'skill_id', 'log_id': 'order_id'})
    df['skill_id'] = df['skill_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    
    # تحويل عمود الإجابة لـ (1 للصح) و (0 للغلط) بشكل صارم
    df['correct'] = (df['correct'] >= 1.0).astype(int)
    
    # خطوة ج: حساب صعوبة المهارة إحصائياً من الداتا (1 - نسبة الحل الصح الإجمالي للمهارة)
    print("⏳ جاري حساب صعوبة كل مهارة إحصائياً...")
    global_difficulty = df.groupby('skill_id')['correct'].mean().to_dict()
    for k in global_difficulty:
        global_difficulty[k] = round(1.0 - global_difficulty[k], 4)
        
    # حقن عمود الصعوبة في الداتا، ولو في مهارة ملهاش داتا بنديها صعوبة متوسطة 0.5
    df['skill_difficulty_avg'] = df['skill_id'].map(global_difficulty).fillna(0.5)
    
    # خطوة د: الترتيب الزمني الصارم (حسب الطالب أولاً ثم ترتيب حل الأسئلة)
    df = df.sort_values(by=['user_id', 'order_id']).reset_index(drop=True)
    
    X_sequences = []
    y_sequences = []
    
    # تقسيم البيانات وتجميعها على مستوى كل طالب لوحده
    grouped = df.groupby('user_id')
    
    print("⏳ جاري تقطيع مصفوفات الطلاب وبناء السلاسل الزمنية المتتابعة...")
    for user_id, group in grouped:
        corrects = group['correct'].values
        difficulties = group['skill_difficulty_avg'].values
        
        # الـ LSTM بيتعلم من التتابع، لو الطالب حل سؤال واحد بس في تاريخه كله بنعمله تخطي
        if len(corrects) < 2:
            continue
            
        # دمج ميزتين مع بعض لكل خطوة حل [الرد الحالي، صعوبة السؤال الحالي]
        features = np.column_stack((corrects, difficulties))
        
        # هندسة ميزات تتبع المعرفة العميقة (Deep Knowledge Tracing):
        # المدخلات X: من السؤال الأول وحتى السؤال قبل الأخير
        # المستهدف y: إجابات الطالب الفعليه من السؤال الثاني للنهاية (عشان ن予測 السؤال التالي دايماً)
        X_student = features[:-1]
        y_student = corrects[1:]
        
        # تقطيع السلاسل الطويلة جداً لقطع ثابتة بحجم 50 سؤال عشان الـ LSTM ميهنجش ويحفظ بكفاءة
        for i in range(0, len(X_student), max_seq_len):
            X_chunk = X_student[i:i+max_seq_len]
            y_chunk = y_student[i:i+max_seq_len]
            
            # عمل Padding (حشو بالـ -1.0) لو الطالب حل أقل من 50 سؤال في السلسلة دي
            if len(X_chunk) < max_seq_len:
                pad_width = max_seq_len - len(X_chunk)
                # الحشو بـ -1.0 بيخلي الـ LSTM يتجاهل الخانات دي تماماً أثناء التدريب (Masking)
                X_chunk = np.pad(X_chunk, ((0, pad_width), (0, 0)), mode='constant', constant_values=-1.0)
                y_chunk = np.pad(y_chunk, (0, pad_width), mode='constant', constant_values=-1.0)
                
            X_sequences.append(X_chunk)
            y_sequences.append(y_chunk)

    # تحويل القوائم إلى مصفوفات Numpy جاهزة تماماً للتعلم العميق
    X_final = np.array(X_sequences, dtype=np.float32)
    y_final = np.array(y_sequences, dtype=np.float32)
    
    # خطوة هـ: حفظ الملفات النهائية على الجهاز
    os.makedirs(os.path.dirname(X_OUTPUT_PATH), exist_ok=True)
    np.save(X_OUTPUT_PATH, X_final)
    np.save(Y_OUTPUT_PATH, y_final)
    
    # حفظ البيانات الوصفية (Metadata) عشان الـ API يستعين بيها في الصعوبات بعدين
    metadata = {
        'max_seq_len': max_seq_len,
        'global_difficulty': global_difficulty,
        'features_count': X_final.shape[2]  # هتكون قيمتها 2 [الرد، الصعوبة]
    }
    joblib.dump(metadata, METADATA_PATH)
    
    print("\n" + "="*60)
    print("✅ تم الانتهاء من معالجة البيانات للـ LSTM بنجاح!")
    print(f"📦 تم حفظ مصفوفة المدخلات (X) في -> {X_OUTPUT_PATH} | الأبعاد: {X_final.shape}")
    print(f"📦 تم حفظ مصفوفة المستهدف (y) في -> {Y_OUTPUT_PATH} | الأبعاد: {y_final.shape}")
    print(f"⚙️ تم حفظ البيانات الوصفية والـ صعوبات في -> {METADATA_PATH}")
    print("="*60)

if __name__ == "__main__":
    preprocess_for_lstm(DEFAULT_INPUT_FILE, max_seq_len=50)