import pandas as pd
import os
import sys

output_path = r"c:\Users\aless\Desktop\IFCS\predictions.csv"
test_path = r"c:\Users\aless\Desktop\IFCS\test_features.csv"

def main():
    print("=== Avvio Verifica Formale di predictions.csv ===")
    
    if not os.path.exists(output_path):
        print(f"ERRORE: Il file {output_path} non esiste!")
        sys.exit(1)
        
    if not os.path.exists(test_path):
        print(f"ERRORE: Impossibile trovare il test set originale ({test_path}) per il confronto delle righe.")
        sys.exit(1)
        
    # Caricamento
    try:
        df_pred = pd.read_csv(output_path, dtype={'pred_class': str})
        df_test = pd.read_csv(test_path)
    except Exception as e:
        print(f"ERRORE durante la lettura dei file: {e}")
        sys.exit(1)
        
    errors = 0
    
    # 1. Verifica colonne
    expected_cols = ['Company ID', 'pred_class']
    if list(df_pred.columns) != expected_cols:
        print(f"ERRORE: Le colonne non corrispondono. Attese {expected_cols}, trovate {list(df_pred.columns)}")
        errors += 1
    else:
        print("OK: Le colonne corrispondono esattamente a ['Company ID', 'pred_class']")
        
    # 2. Verifica numero di righe
    expected_rows = len(df_test)
    found_rows = len(df_pred)
    if found_rows != expected_rows:
        print(f"ERRORE: Il numero di righe non corrisponde. Attese {expected_rows} righe (dal test set), trovate {found_rows}")
        errors += 1
    else:
        print(f"OK: Il numero di righe corrisponde esattamente al test set ({expected_rows} righe)")
        
    # 3. Verifica valori unici in pred_class
    unique_vals = set(df_pred['pred_class'].unique())
    allowed_vals = {'TRUE', 'FALSE'}
    # Check if there are other values
    invalid_vals = unique_vals - allowed_vals
    if invalid_vals:
        print(f"ERRORE: Sono presenti valori non ammessi in pred_class: {invalid_vals}. Sono ammessi solo 'TRUE' o 'FALSE' (maiuscoli)")
        errors += 1
    else:
        print("OK: I valori in pred_class sono esclusivamente 'TRUE' o 'FALSE'")
        
    # 4. Verifica valori nulli
    null_counts = df_pred.isnull().sum().sum()
    if null_counts > 0:
        print(f"ERRORE: Ci sono {null_counts} valori nulli nel file!")
        errors += 1
    else:
        print("OK: Nessun valore nullo presente")
        
    # 5. Verifica corrispondenza ID azienda
    test_ids = set(df_test['Company ID'])
    pred_ids = set(df_pred['Company ID'])
    if test_ids != pred_ids:
        print(f"ERRORE: Gli ID delle aziende non corrispondono perfettamente tra il test set e le predizioni!")
        errors += 1
    else:
        print("OK: Tutti gli ID azienda del test set sono presenti e corrispondono perfettamente")
        
    print("\n--- Sintesi della Verifica ---")
    if errors == 0:
        print("VERIFICA COMPLETATA CON SUCCESSO! Il file predictions.csv rispetta tutti i requisiti formali del brief.")
        # Stampa le prime righe per controllo visivo
        print("\nAnteprima delle prime 5 righe:")
        print(df_pred.head())
    else:
        print(f"VERIFICA FALLITA con {errors} errore/i. Si prega di correggere prima della sottomissione.")
        sys.exit(1)

if __name__ == '__main__':
    main()
