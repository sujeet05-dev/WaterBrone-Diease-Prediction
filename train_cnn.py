import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate, Embedding, Dropout
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), 'model', '2_waterborne_diseases_lab_10k_clean.csv')

def build_cnn(max_words, max_len, num_features, num_classes):
    # Text input branch
    text_input = Input(shape=(max_len,), name='text_input')
    x = Embedding(input_dim=max_words, output_dim=64, input_length=max_len)(text_input)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)

    # Numerical/Categorical input branch
    dense_input = Input(shape=(num_features,), name='dense_input')
    y = Dense(64, activation='relu')(dense_input)
    y = Dropout(0.3)(y)
    y = Dense(32, activation='relu')(y)

    # Combine branches
    concat = concatenate([x, y])
    z = Dense(64, activation='relu')(concat)
    z = Dropout(0.3)(z)
    output = Dense(num_classes, activation='softmax', name='output')(z)

    model = Model(inputs=[text_input, dense_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preprocessing text...")
    symptoms = df['Symptoms_Text'].astype(str).tolist()
    
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(symptoms)
    
    seq = tokenizer.texts_to_sequences(symptoms)
    max_len = 60 # To match main.py
    X_text = pad_sequences(seq, maxlen=max_len, padding='post')
    
    print("Preprocessing dense features...")
    num_cols = ['Sodium_mmol_L', 'Potassium_mmol_L', 'Chloride_mmol_L', 'WBC_109_per_L', 
                'Hemoglobin_g_dL', 'Platelets_109_per_L', 'Urea_mg_dL', 'Creatinine_mg_dL', 
                'Bilirubin_mg_dL', 'ALT_U_L', 'AST_U_L', 'Age', 'Hygiene_Score']

    df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
    df['Water_Bottled'] = (df['Water_Source'] == 'Bottled').astype(int)
    df['Water_River'] = (df['Water_Source'] == 'River').astype(int)
    df['Water_Well'] = (df['Water_Source'] == 'Well').astype(int)

    cat_cols = ['Gender_Male', 'Water_Bottled', 'Water_River', 'Water_Well']

    X_num = df[num_cols].values
    X_cat = df[cat_cols].values
    
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    X_dense = np.hstack([X_num_scaled, X_cat])
    
    print("Preprocessing target...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['Disease'])
    num_classes = len(encoder.classes_)
    
    print("Splitting data...")
    X_text_train, X_text_test, X_dense_train, X_dense_test, y_train, y_test = train_test_split(
        X_text, X_dense, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Building model...")
    model = build_cnn(max_words, max_len, X_dense.shape[1], num_classes)
    model.summary()
    
    print("Training model...")
    history = model.fit(
        [X_text_train, X_dense_train], y_train,
        validation_data=([X_text_test, X_dense_test], y_test),
        epochs=10, 
        batch_size=32
    )

    print("Saving model and preprocessors...")
    os.makedirs('model/cnn_output', exist_ok=True)
    model.save('model/cnn_output/best_cnn_model.h5')
    joblib.dump(tokenizer, 'model/cnn_output/tokenizer_cnn.joblib')
    joblib.dump(scaler, 'model/cnn_output/scaler_cnn.joblib')
    joblib.dump(encoder, 'model/cnn_output/label_encoder_cnn.joblib')
    
    print("Training Complete. Model and preprocessors saved.")
    
if __name__ == '__main__':
    main()
