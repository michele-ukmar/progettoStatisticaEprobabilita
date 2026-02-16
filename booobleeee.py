import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def carica_dataset(nome_file):
    try:
        df = pd.read_csv(nome_file)
        print(f"[OK] Dataset caricato: {len(df)} mazzi analizzati.")
        return df
    except FileNotFoundError:
        print(f"[ERRORE] Il file '{nome_file}' non esiste. Controlla la cartella.")
        exit()

def genera_report_testuale(df):
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

    print(f"\n1b. ANALISI DELLA FORMA (Curtosi e Asimmetria)")
    print(f"   - Curtosi Cycle:    {stats.kurtosis(cycle):.2f}")
    print(f"   - Curtosi Beatdown: {stats.kurtosis(beatdown):.2f}")
    print("   (Nota: Curtosi alta = curva più 'appuntita', risultati molto concentrati)")

    # 2. Correlazione (Relazione Lineare)
    corr = df['Costo_Elisir'].corr(df['Win_Rate'])
    cov = np.cov(df['Costo_Elisir'], df['Win_Rate'])[0][1]
    
    print(f"\n2. RELAZIONE COSTO-VITTORIA")
    print(f"   - Covarianza:   {cov:.4f}")
    print(f"   - Correlazione: {corr:.4f}")
    
    if abs(corr) < 0.3:
        print("   -> INTERPRETAZIONE: Non esiste una correlazione significativa.")
        print("      Vincere non dipende dal costo del mazzo (Skill > Elisir).")

def calcola_bayes(df):
    top_tier = df[df['Win_Rate'] > 52.0]
    prob_top_tier = len(top_tier) / len(df)
    beatdown_top = top_tier[top_tier['Categoria'] == 'Beatdown']
    prob_intersezione = len(beatdown_top) / len(df)
    
    if prob_top_tier > 0:
        prob_condizionata = prob_intersezione / prob_top_tier
        print(f"\n3. PROBABILITÀ CONDIZIONATA (BAYES)")
        print(f"   - Probabilità che un mazzo Top Tier sia Beatdown: {prob_condizionata*100:.1f}%")

def mostra_grafici(df):
    """Genera dashboard con Bubble Chart e Gaussiana."""
    plt.figure(figsize=(16, 10))

    # GRAFICO 1: Box Plot
    plt.subplot(2, 2, 1)
    sns.boxplot(x='Categoria', y='Win_Rate', data=df, palette="Set2")
    plt.title('1. Variabilità Win Rate (Box Plot)')
    plt.grid(True, alpha=0.3)

    # --- GRAFICO 2: BUBBLE CHART (Costo vs WinRate pesato per Partite) ---
    plt.subplot(2, 2, 2)
    # Usiamo scatterplot per gestire la dimensione (size) dei punti in base alle 'Partite'
    sns.scatterplot(data=df, x='Costo_Elisir', y='Win_Rate', 
                    hue='Categoria', size='Partite', sizes=(40, 500), 
                    alpha=0.6, palette="Set1")
    # Aggiungiamo la linea di regressione sopra
    sns.regplot(data=df, x='Costo_Elisir', y='Win_Rate', scatter=False, color='black', line_kws={'linestyle':'--'})
    plt.title('2. Costo vs Vittoria (Dimensione = Popolarità)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # GRAFICO 3: DISTRIBUZIONE GAUSSIANA
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='Win_Rate', hue='Categoria', kde=True, element="step", stat="density", common_norm=False)
    plt.title('3. Analisi della Distribuzione (Gaussiana)')
    plt.ylabel('Densità di Probabilità')
    
    # GRAFICO 4: Volume di Gioco Totale
    plt.subplot(2, 2, 4)
    # Mostriamo la somma totale delle partite per categoria per evidenziare lo sbilanciamento di uso
    sns.barplot(x='Categoria', y='Partite', data=df, estimator=np.sum, palette="viridis", errorbar=None)
    plt.title('4. Volume Totale di Partite per Categoria')
    plt.ylabel('Somma Partite Giocate')

    print("\n[INFO] Generazione dashboard... Chiudi la finestra del grafico per terminare.")
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'dataset_clash.csv'
    dataset = carica_dataset(file_path)
    genera_report_testuale(dataset)
    calcola_bayes(dataset)
    mostra_grafici(dataset)

if __name__ == "__main__":
    main()