import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- FUNZIONE 1: Caricamento Dati ---
def carica_dataset(nome_file):
    """Carica il CSV e gestisce eventuali errori."""
    try:
        df = pd.read_csv(nome_file)
        print(f"[OK] Dataset caricato: {len(df)} mazzi analizzati.")
        return df
    except FileNotFoundError:
        print(f"[ERRORE] Il file '{nome_file}' non esiste. Controlla la cartella.")
        exit()

# --- FUNZIONE 2: Calcoli Statistici ---
def genera_report_testuale(df):
    """Calcola e stampa le statistiche descrittive (Media, Varianza, Correlazione)."""
    cycle = df[df['Categoria'] == 'Cycle']['Win_Rate']
    beatdown = df[df['Categoria'] == 'Beatdown']['Win_Rate']

    print("\n" + "="*50)
    print(" REPORT ANALISI: CYCLE vs BEATDOWN")
    print("="*50)

    # 1. Analisi Variabilità (Rischio)
    print(f"\n1. ANALISI DEL RISCHIO (Deviazione Standard)")
    print(f"   - Cycle (Mazzi Leggeri):     {cycle.std():.2f} (Stabilità: Alta)")
    print(f"   - Beatdown (Mazzi Pesanti):  {beatdown.std():.2f} (Stabilità: Bassa)")
    
    if beatdown.std() > cycle.std():
        print("   -> CONCLUSIONE: I mazzi Beatdown sono più 'rischiosi' e incostanti.")
    else:
        print("   -> CONCLUSIONE: La variabilità è simile tra le categorie.")

    # 2. Correlazione (Relazione Lineare)
    corr = df['Costo_Elisir'].corr(df['Win_Rate'])
    cov = np.cov(df['Costo_Elisir'], df['Win_Rate'])[0][1]
    
    print(f"\n2. RELAZIONE COSTO-VITTORIA")
    print(f"   - Covarianza:   {cov:.4f}")
    print(f"   - Correlazione: {corr:.4f}")
    
    if abs(corr) < 0.3:
        print("   -> INTERPRETAZIONE: Non esiste una correlazione significativa.")
        print("      Vincere non dipende da quanto costa il mazzo (Skill > Costo).")
    elif corr > 0:
        print("   -> INTERPRETAZIONE: Correlazione Positiva (Mazzi costosi avvantaggiati).")

# --- FUNZIONE 3: Probabilità Bayesiana ---
def calcola_bayes(df):
    """Esempio di Probabilità Condizionata: P(Beatdown | WinRate > 52%)."""
    # Definiamo 'Top Tier' i mazzi con Win Rate > 52%
    top_tier = df[df['Win_Rate'] > 52.0]
    
    # Probabilità Totale di essere Top Tier P(B)
    prob_top_tier = len(top_tier) / len(df)
    
    # Probabilità Congiunta P(A e B) -> Essere Beatdown E Top Tier
    beatdown_top = top_tier[top_tier['Categoria'] == 'Beatdown']
    prob_intersezione = len(beatdown_top) / len(df)
    
    # Teorema di Bayes (semplificato): P(A|B) = P(A intersecato B) / P(B)
    if prob_top_tier > 0:
        prob_condizionata = prob_intersezione / prob_top_tier
        print(f"\n3. PROBABILITÀ CONDIZIONATA (BAYES)")
        print(f"   - Domanda: Se vedo un mazzo vincente (>52%), che probabilità c'è che sia Beatdown?")
        print(f"   - Risposta: {prob_condizionata*100:.1f}%")
    else:
        print("\n3. PROBABILITÀ: Nessun mazzo supera la soglia del 52%.")

# --- FUNZIONE 4: Grafici ---
def mostra_grafici(df):
    """Genera Box Plot e Scatter Plot."""
    plt.figure(figsize=(14, 6))

    # Grafico Sinistra: Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Categoria', y='Win_Rate', data=df, palette="Set2")
    plt.title('Variabilità dei Risultati (Rischio)')
    plt.ylabel('Win Rate %')
    plt.grid(True, alpha=0.3)

    # Grafico Destra: Scatter Plot
    plt.subplot(1, 2, 2)
    sns.regplot(x='Costo_Elisir', y='Win_Rate', data=df, line_kws={'color':'red'})
    plt.title(f'Correlazione Costo vs Vittoria')
    plt.xlabel('Costo Elisir')
    plt.grid(True, alpha=0.3)

    print("\n[INFO] Generazione grafici in corso...")
    plt.tight_layout()
    plt.show()

# --- IL MAIN (Punto di ingresso) ---
def main():
    # 1. Configurazione
    file_path = 'dataset_clash.csv'
    
    # 2. Esecuzione sequenziale
    dataset = carica_dataset(file_path)
    genera_report_testuale(dataset)
    calcola_bayes(dataset)
    mostra_grafici(dataset)

# Questo controllo serve a dire a Python: "Esegui main solo se lancio questo file direttamente"
if __name__ == "__main__":
    main()