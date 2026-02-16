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
    """
    Genera dashboard professionale con Box Plot al posto del Violin Plot.
    """
    
    # 1. IMPOSTAZIONI ESTETICHE
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['figure.figsize'] = (20, 14)
    plt.rcParams['font.weight'] = 'bold'
    
    # Palette fissa: Cycle=Blu, Beatdown=Rosso, Midrange=Verde
    custom_palette = {"Cycle": "#3498db", "Beatdown": "#e74c3c", "Midrange": "#2ecc71"}

    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    plt.suptitle("ANALISI STATISTICA CLASH ROYALE: META-GAME REPORT", fontsize=24, fontweight='bold', y=0.96)

    # --- GRAFICO 1: BUBBLE CHART ---
    ax1 = axs[0, 0]
    sns.scatterplot(data=df, x='Costo_Elisir', y='Win_Rate', 
                    hue='Categoria', palette=custom_palette,
                    size='Partite', sizes=(50, 600), 
                    alpha=0.6, edgecolor="black", linewidth=1, ax=ax1)
    
    sns.regplot(data=df, x='Costo_Elisir', y='Win_Rate', scatter=False, 
                color='#2c3e50', line_kws={'linestyle':'--', 'linewidth': 2}, ax=ax1)
    
    ax1.set_title('1. Costo vs Win Rate (Dimensione = Popolarità)', fontsize=16)
    ax1.set_xlabel('Costo Elisir Medio')
    ax1.set_ylabel('Win Rate %')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    # --- GRAFICO 2: GAUSSIANE (KDE) ---
    ax2 = axs[0, 1]
    sns.kdeplot(data=df, x='Win_Rate', hue='Categoria', palette=custom_palette, 
                fill=True, alpha=0.3, linewidth=3, common_norm=False, ax=ax2)
    
    ax2.set_title('2. Distribuzioni di Probabilità (Gaussiane)', fontsize=16)
    ax2.set_xlabel('Win Rate %')
    ax2.set_ylabel('Densità')

    # --- GRAFICO 3: BOX PLOT (TORNATO!) ---
    ax3 = axs[1, 0]
    # width=0.5 rende i box più snelli ed eleganti
    sns.boxplot(x='Categoria', y='Win_Rate', data=df, palette=custom_palette, 
                width=0.5, linewidth=2, fliersize=5, ax=ax3)
    
    ax3.set_title('3. Analisi dei Quartili e Outliers (Box Plot)', fontsize=16)
    ax3.set_ylabel('Win Rate %')
    ax3.set_xlabel('')

    # --- GRAFICO 4: VOLUME TOTALE ---
    ax4 = axs[1, 1]
    barplot = sns.barplot(x='Categoria', y='Partite', data=df, estimator=np.sum, 
                          palette=custom_palette, errorbar=None, ax=ax4, edgecolor="black")
    
    ax4.set_title('4. Volume Totale Partite Giocate', fontsize=16)
    ax4.set_ylabel('Totale Partite')
    ax4.set_xlabel('')
    
    # Numeri sulle barre
    for p in barplot.patches:
        height = p.get_height()
        ax4.text(p.get_x() + p.get_width()/2., height + 1000,
                f'{int(height):,}', ha="center", fontsize=14, fontweight='bold', color='#2c3e50')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    sns.despine()
    print("\n[INFO] Generazione Dashboard con Box Plot...")
    plt.show()

def main():
    file_path = 'dataset_clash.csv'
    dataset = carica_dataset(file_path)
    genera_report_testuale(dataset)
    calcola_bayes(dataset)
    mostra_grafici(dataset)

if __name__ == "__main__":
    main()