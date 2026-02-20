import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

COLONNE_OBBLIGATORIE = {"Categoria", "Win_Rate", "Costo_Elisir", "Partite"}


def carica_dataset(nome_file):
    try:
        df = pd.read_csv(nome_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Il file '{nome_file}' non esiste. Controlla la cartella."
        ) from exc

    print(f"[OK] Dataset caricato: {len(df)} mazzi analizzati.")
    return df


def valida_dataset(df):
    missing = COLONNE_OBBLIGATORIE - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {sorted(missing)}")

    df = df.dropna(subset=["Categoria", "Win_Rate", "Costo_Elisir", "Partite"])
    if df.empty:
        raise ValueError("Il dataset e' vuoto dopo la pulizia dei dati mancanti.")
    return df

def genera_report_testuale(df):
    cycle = df[df['Categoria'] == 'Cycle']['Win_Rate']
    beatdown = df[df['Categoria'] == 'Beatdown']['Win_Rate']

    print("\n" + "="*50)
    print(" REPORT ANALISI: CYCLE vs BEATDOWN")
    print("="*50)

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

    corr = df['Costo_Elisir'].corr(df['Win_Rate'])
    validi = df[["Costo_Elisir", "Win_Rate"]].dropna()
    cov = np.cov(validi["Costo_Elisir"], validi["Win_Rate"])[0][1] if not validi.empty else np.nan
    
    print(f"\n2. RELAZIONE COSTO-WIN RATE")
    print(f"   - Covarianza:   {cov:.4f}")
    print(f"   - Correlazione: {corr:.4f}")
    
    if abs(corr) < 0.3:
        print("   -> INTERPRETAZIONE: Non esiste una correlazione significativa.")
        print("      Vincere non dipende dal costo del mazzo (Skill > Elisir).")

def calcola_bayes(df, soglia_top=52.0):
    top_tier = df[df['Win_Rate'] > soglia_top]
    prob_top_tier = len(top_tier) / len(df)
    beatdown_top = top_tier[top_tier['Categoria'] == 'Beatdown']
    prob_intersezione = len(beatdown_top) / len(df)
    
    if prob_top_tier > 0:
        prob_condizionata = prob_intersezione / prob_top_tier
        print(f"\n3. PROBABILITÀ CONDIZIONATA (BAYES)")
        print(f"   - Probabilità che un mazzo di fascia alta sia Beatdown: {prob_condizionata*100:.1f}%")

def mostra_grafici(df):
    """Genera dashboard con 4 grafici."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.weight'] = 'bold'
    
    custom_palette = {"Cycle": "#3498db", "Beatdown": "#e74c3c", "Midrange": "#2ecc71"}

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("ANALISI STATISTICA CLASH ROYALE: REPORT DEL META-GAME", fontsize=18, fontweight='bold')

    ax1 = axs[0, 0]
    sns.scatterplot(data=df, x='Costo_Elisir', y='Win_Rate', 
                    hue='Categoria', palette=custom_palette,
                    size='Partite', sizes=(50, 600), 
                    alpha=0.6, edgecolor="black", linewidth=1, ax=ax1)
    
    sns.regplot(data=df, x='Costo_Elisir', y='Win_Rate', scatter=False, 
                color='#2c3e50', line_kws={'linestyle':'--', 'linewidth': 2}, ax=ax1)
    
    ax1.set_title('1. Costo vs Win Rate (Dimensione = Popolarità)', fontsize=12, pad=8)
    ax1.set_xlabel('Costo Elisir Medio', fontsize=10)
    ax1.set_ylabel('Win Rate %', fontsize=10)
    ax1.get_legend().remove()

    ax2 = axs[0, 1]
    sns.kdeplot(data=df, x='Win_Rate', hue='Categoria', palette=custom_palette, 
                fill=True, alpha=0.3, linewidth=3, common_norm=False, ax=ax2)
    
    ax2.set_title('2. Distribuzioni di Probabilità (Gaussiane)', fontsize=12, pad=8)
    ax2.set_xlabel('Win Rate %', fontsize=10)
    ax2.set_ylabel('Densità', fontsize=10)
    legend = ax2.get_legend()
    legend.set_title('Categoria')
    legend.set_bbox_to_anchor((0.98, 0.98))
    legend.set_loc('upper right')

    ax3 = axs[1, 0]
    sns.boxplot(x='Categoria', y='Win_Rate', data=df, palette=custom_palette, 
                width=0.5, linewidth=2, fliersize=5, ax=ax3)
    
    ax3.set_title('3. Analisi dei Quartili e Valori Anomali (Box Plot)', fontsize=12, pad=8)
    ax3.set_ylabel('Win Rate %', fontsize=10)
    ax3.set_xlabel('')

    ax4 = axs[1, 1]
    barplot = sns.barplot(x='Categoria', y='Partite', data=df, estimator=np.sum, 
                        palette=custom_palette, errorbar=None, ax=ax4, edgecolor="black")
    
    ax4.set_title('4. Volume Totale Partite Giocate', fontsize=12, pad=8)
    ax4.set_ylabel('Totale Partite', fontsize=10)
    ax4.set_xlabel('')
    
    for p in barplot.patches:
        height = p.get_height()
        ax4.text(p.get_x() + p.get_width()/2., height + 1000,
                f'{int(height):,}', ha="center", fontsize=10, fontweight='bold', color='#2c3e50')

    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.93, hspace=0.30, wspace=0.28)
    sns.despine()
    print("\n[INFO] Generazione dashboard con box plot...")
    plt.show()

def main():
    file_path = 'dataset_clash.csv'
    dataset = carica_dataset(file_path)
    dataset = valida_dataset(dataset)
    genera_report_testuale(dataset)
    calcola_bayes(dataset)
    mostra_grafici(dataset)

if __name__ == "__main__":
    main()