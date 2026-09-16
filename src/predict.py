import pandas as pd
import numpy as np
import pickle
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

test_path = r"c:\Users\aless\Desktop\IFCS\test_features.csv"
model_pipeline_path = r"c:\Users\aless\Desktop\IFCS\logistic_model_pipeline.pkl"
output_path = r"c:\Users\aless\Desktop\IFCS\predictions.csv"

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

def main():
    # Set up argument parsing to allow choosing the threshold method dynamically
    parser = argparse.ArgumentParser(description="Task B Inference Script")
    parser.add_argument(
        "--method", 
        type=str, 
        default="max_f1", 
        choices=["max_f1", "youden_j", "min_distance"],
        help="Criterio di ottimizzazione della soglia: max_f1 (default), youden_j, o min_distance"
    )
    args = parser.parse_args()

    print("=== Avvio Generazione Predizioni Task B ===")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Impossibile trovare il file di test in: {test_path}")
    if not os.path.exists(model_pipeline_path):
        raise FileNotFoundError(f"Impossibile trovare la pipeline del modello in: {model_pipeline_path}. Esegui prima train_model.py!")

    # 1. Caricamento della pipeline
    print(f"Caricamento della pipeline da: {model_pipeline_path}")
    with open(model_pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    model = pipeline['model']
    scaler = pipeline['scaler']
    features = pipeline['features']
    log_all = pipeline['log_all']
    thresholds = pipeline['thresholds']
    
    # Seleziona la soglia
    selected_method = args.method
    threshold = thresholds[selected_method]
    print(f"Metodo soglia selezionato: {selected_method}")
    print(f"Valore soglia applicato: {threshold:.4f}")

    # 2. Caricamento dati di test
    print(f"Caricamento del dataset di test da: {test_path}")
    df_test = pd.read_csv(test_path, sep=',', encoding='latin-1')
    
    # 3. Estrazione e pre-elaborazione delle feature
    print("Pre-elaborazione delle feature sul test set...")
    X_test = df_test[features].copy()
    
    # Applica signed log-transform coerentemente con l'addestramento
    if log_all:
        for col in features:
            X_test[col] = signed_log1p(X_test[col])
    else:
        X_test['Total financial expenses'] = signed_log1p(X_test['Total financial expenses'])
        
    # Applica lo scaler pre-addestrato
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Predizione delle probabilità
    print("Calcolo delle probabilità di dissesto...")
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Applica la soglia
    preds = (probs >= threshold).astype(int)
    
    # Mappatura delle predizioni in TRUE/FALSE (stringhe maiuscole) come richiesto dal brief
    pred_classes = np.where(preds == 1, 'TRUE', 'FALSE')
    
    # 5. Creazione del dataframe di output
    df_output = pd.DataFrame({
        'Company ID': df_test['Company ID'],
        'pred_class': pred_classes
    })
    
    # Verifica formale dell'output prima del salvataggio
    assert df_output.shape[0] == df_test.shape[0], "Il numero di righe predette non corrisponde al test set!"
    assert list(df_output.columns) == ['Company ID', 'pred_class'], "Colonne di output non corrette!"
    
    # Salvataggio
    df_output.to_csv(output_path, index=False)
    print(f"Predizioni salvate con successo in: {output_path}")
    
    # Statistiche finali
    n_total = len(df_output)
    n_distressed = (df_output['pred_class'] == 'TRUE').sum()
    pct_distressed = (n_distressed / n_total) * 100
    
    print("\n--- Statistiche delle Predizioni ---")
    print(f"Totale aziende nel test set: {n_total}")
    print(f"Aziende predette in dissesto (TRUE): {n_distressed} ({pct_distressed:.2f}%)")
    print(f"Aziende predette sane (FALSE): {n_total - n_distressed} ({100 - pct_distressed:.2f}%)")

if __name__ == '__main__':
    main()
