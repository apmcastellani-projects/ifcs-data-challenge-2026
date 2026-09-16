import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Configurazione del run
LOG_ALL_FEATURES = True  # Raccomandato: True (applica log con segno a tutte e 4 le variabili)
                         # Impostare a False per applicarlo SOLO a 'Total financial expenses'

train_path = r"c:\Users\aless\Desktop\IFCS\train_cleaned_final_ORIG.csv"
model_pipeline_path = r"c:\Users\aless\Desktop\IFCS\logistic_model_pipeline.pkl"

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

def get_features(df_orig, log_all=True):
    raw_cols = ['Sales Revenue', 'Employees', 'Operating Income', 'Total financial expenses']
    X = df_orig[raw_cols].copy()
    if log_all:
        for col in raw_cols:
            X[col] = signed_log1p(X[col])
    else:
        # Applica solo a Total financial expenses
        X['Total financial expenses'] = signed_log1p(X['Total financial expenses'])
    return X, raw_cols

def main():
    print("=== Avvio Addestramento Modello Task B ===")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Impossibile trovare il file di train in: {train_path}")
        
    # 1. Caricamento dati
    print(f"Caricamento del dataset da: {train_path}")
    df = pd.read_csv(train_path, sep=';', decimal=',', encoding='latin-1')
    y = df['Financial distress'].astype(int).values
    
    # Estrazione feature
    X, features = get_features(df, log_all=LOG_ALL_FEATURES)
    print(f"Features selezionate: {features}")
    print(f"Trasformazione signed_log1p applicata a tutte le feature? {LOG_ALL_FEATURES}")
    print(f"Dimensioni del dataset: {X.shape}")
    
    # 2. Cross-Validation per stima soglia ottimale ed evitare Data Leakage
    print("\nEsecuzione 5-Fold Stratified Cross-Validation per stima soglia decisionale...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_va = y[train_idx], y[val_idx]
        
        scaler_fold = StandardScaler()
        X_tr_scaled = scaler_fold.fit_transform(X_tr)
        X_va_scaled = scaler_fold.transform(X_va)
        
        model_fold = LogisticRegression(random_state=42, max_iter=1000)
        model_fold.fit(X_tr_scaled, y_tr)
        oof_probs[val_idx] = model_fold.predict_proba(X_va_scaled)[:, 1]
        
    cv_auc = roc_auc_score(y, oof_probs)
    print(f"CV ROC AUC: {cv_auc:.5f}")
    
    # Ricerca soglie ottimali
    # A. Max F1-Score
    best_f1_score = 0
    best_f1_thresh = 0.5
    for t in np.linspace(0.01, 0.99, 99):
        preds = (oof_probs >= t).astype(int)
        f = f1_score(y, preds)
        if f > best_f1_score:
            best_f1_score = f
            best_f1_thresh = t
            
    # B. Youden's J Index (Max TPR - FPR)
    fpr, tpr, thresholds = roc_curve(y, oof_probs)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_j_thresh = thresholds[best_j_idx]
    
    # C. Min Distance to (0,1)
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    best_d_idx = np.argmin(distances)
    best_d_thresh = thresholds[best_d_idx]
    
    print("\n--- Analisi Soglie Ottimali su OOF predictions ---")
    
    # Funzione di utilità per stampare report
    def print_metrics(t, name):
        preds = (oof_probs >= t).astype(int)
        f1 = f1_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        cm = confusion_matrix(y, preds)
        print(f"Opzione: {name}")
        print(f"  Soglia: {t:.4f}")
        print(f"  F1-Score: {f1:.5f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
        print(f"  Confusion Matrix:\n  {cm[0]}\n  {cm[1]}")
        print()
        
    print_metrics(best_f1_thresh, "Massimo F1-Score")
    print_metrics(best_j_thresh, "Indice J di Youden (Max TPR - FPR)")
    print_metrics(best_d_thresh, "Distanza Minima da (0,1)")
    
    # 3. Addestramento finale su TUTTO il dataset di train
    print("Addestramento finale sul dataset completo...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    model_final = LogisticRegression(random_state=42, max_iter=1000)
    model_final.fit(X_scaled, y)
    
    # 4. Salvataggio della pipeline
    pipeline = {
        'model': model_final,
        'scaler': scaler_final,
        'features': features,
        'log_all': LOG_ALL_FEATURES,
        'thresholds': {
            'max_f1': best_f1_thresh,
            'youden_j': best_j_thresh,
            'min_distance': best_d_thresh
        },
        'default_threshold_method': 'max_f1'  # Metodo di default
    }
    
    with open(model_pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline del modello salvata con successo in: {model_pipeline_path}")
    print("Pronta per l'uso nello script di predizione.")

if __name__ == '__main__':
    main()
