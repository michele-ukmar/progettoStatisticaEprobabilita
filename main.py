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
    
    # Remove invalid category rows (header rows or bad data)
    df = df[df['Categoria'] != 'Category']
    
    # Convert numeric columns to proper types
    df['Win_Rate'] = pd.to_numeric(df['Win_Rate'], errors='coerce')
    df['Costo_Elisir'] = pd.to_numeric(df['Costo_Elisir'], errors='coerce')
    df['Partite'] = pd.to_numeric(df['Partite'], errors='coerce')
    
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

def grafico_top_5_combinato(df):
    """Genera un unico grafico con tutti i top 5 mazzi (winrate + partite) combinati con assi sincronizzati."""
    from matplotlib.patches import Patch
    
    # Palette coerente
    custom_palette = {"Cycle": "#3498db", "Beatdown": "#e74c3c", "Midrange": "#2ecc71"}
    
    # Seleziona i top 5 mazzi per winrate e partite
    df_top5_wr = df.nlargest(5, 'Win_Rate')[['Mazzo', 'Win_Rate', 'Partite', 'Categoria']].reset_index(drop=True)
    df_top5_pt = df.nlargest(5, 'Partite')[['Mazzo', 'Win_Rate', 'Partite', 'Categoria']].reset_index(drop=True)
    
    # Aggiungi etichetta per identificare il gruppo
    df_top5_wr['Gruppo'] = 'Mazzi più Vincenti'
    df_top5_pt['Gruppo'] = 'Mazzi più Giocati'
    
    # Combina i due dataset
    df_combined = pd.concat([df_top5_wr, df_top5_pt], ignore_index=True)
    
    # Calcola i range sincronizzati per gli assi Y
    max_wr = max(df_top5_wr['Win_Rate'].max(), df_top5_pt['Win_Rate'].max())
    max_pt = max(df_top5_wr['Partite'].max(), df_top5_pt['Partite'].max())
    
    # Configura lo stile
    sns.set_style("whitegrid")
    fig, ax_main = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax_main.set_facecolor('#ffffff')
    
    # Colori per le barre (basati sulla categoria)
    colors = [custom_palette[cat] for cat in df_combined['Categoria']]
    x = np.arange(len(df_combined))
    width = 0.36
    
    # Barre per Win Rate (asse sinistro)
    bars_wr = ax_main.bar(x - width/2, df_combined['Win_Rate'], width, label='Win Rate %', 
                          color=colors, alpha=0.88, edgecolor='#2c3e50', linewidth=2.0)
    
    # Asse destro per Partite Giocate
    ax_aux = ax_main.twinx()
    ax_aux.set_facecolor('none')
    bars_pt = ax_aux.bar(x + width/2, df_combined['Partite'], width, label='Partite Giocate', 
                         color='#f39c12', alpha=0.75, edgecolor='#2c3e50', linewidth=2.0)
    
    # Configura asse sinistro (Win Rate)
    ax_main.set_ylabel('Win Rate %', fontsize=12, fontweight='bold', color='#2c3e50', labelpad=10)
    ax_main.set_ylim(44, max_wr + 2.5)
    ax_main.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8, color='#bdc3c7')
    ax_main.tick_params(axis='y', labelsize=10, colors='#2c3e50')
    
    # Configura asse destro (Partite Giocate) - SINCRONIZZATO
    ax_aux.set_ylabel('Partite Giocate', fontsize=12, fontweight='bold', color='#2c3e50', labelpad=10)
    ax_aux.set_ylim(0, max_pt + max_pt * 0.15)
    ax_aux.tick_params(axis='y', labelsize=10, colors='#2c3e50')
    
    # Configura etichette X
    ax_main.set_xticks(x)
    nomi_mazzi = [mazzo[:20] for mazzo in df_combined['Mazzo']]
    ax_main.set_xticklabels(nomi_mazzi, rotation=35, ha='right', fontsize=10, fontweight='bold', color='#2c3e50')
    
    # Aggiungi linea di separazione tra i due gruppi
    ax_main.axvline(x=4.5, color='#7f8c8d', linestyle='--', linewidth=2.5, alpha=0.6, zorder=2)
    
    # Aggiungi etichette dei gruppi
    ax_main.text(2, max_wr + 1.5, 'Mazzi più Vincenti', fontsize=11, fontweight='bold', 
                color='#2c3e50', ha='center', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5))
    ax_main.text(7, max_wr + 1.5, 'Mazzi più Giocati', fontsize=11, fontweight='bold', 
                color='#2c3e50', ha='center', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5))
    
    # Valori sopra le barre (Win Rate)
    for i, (bar, val) in enumerate(zip(bars_wr, df_combined['Win_Rate'])):
        ax_main.text(bar.get_x() + bar.get_width()/2., val + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', 
                    color=colors[i], bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))
    
    # Valori sopra le barre (Partite Giocate)
    for bar, val in zip(bars_pt, df_combined['Partite']):
        ax_aux.text(bar.get_x() + bar.get_width()/2., val + max_pt * 0.01,
                   f'{int(val):,}', ha='center', va='bottom', fontsize=8, fontweight='bold', 
                   color='#e67e22', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))
    
    # Stile degli spine
    sns.despine(ax=ax_main, top=True, right=True)
    ax_aux.spines['top'].set_visible(False)
    ax_aux.spines['left'].set_visible(False)
    ax_aux.spines['right'].set_visible(True)
    ax_aux.spines['right'].set_linewidth(1.5)
    ax_aux.spines['right'].set_color('#2c3e50')
    
    # Titolo
    ax_main.set_title('Top 10 Mazzi: Mazzi più Vincenti vs Mazzi più Giocati (Assi Y Sincronizzati)', 
                     fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    
    # Legenda
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='#2c3e50', label='Beatdown', linewidth=1.5),
        Patch(facecolor='#3498db', edgecolor='#2c3e50', label='Cycle', linewidth=1.5),
        Patch(facecolor='#2ecc71', edgecolor='#2c3e50', label='Midrange', linewidth=1.5),
        Patch(facecolor='#f39c12', edgecolor='#2c3e50', alpha=0.88, label='Partite Giocate', linewidth=1.5),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
              framealpha=0.96, edgecolor='#2c3e50', fancybox=True, shadow=True, 
              bbox_to_anchor=(0.5, -0.08), frameon=True)
    
    plt.subplots_adjust(bottom=0.18, left=0.08, right=0.92)
    fig.savefig('grafico_7_top5_combinato.png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    print("[OK] Grafico combinato Top 10 mazzi (assi sincronizzati) salvato come 'grafico_7_top5_combinato.png'")
    plt.close(fig)

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
                    s=80, alpha=0.6, edgecolor="black", linewidth=1, ax=ax1)
    
    sns.regplot(data=df, x='Costo_Elisir', y='Win_Rate', scatter=False, 
                color='#2c3e50', line_kws={'linestyle':'--', 'linewidth': 2}, ax=ax1)
    
    ax1.set_title('1. Costo vs Win Rate', fontsize=12, pad=8)
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
    sns.boxplot(x='Categoria', y='Win_Rate', data=df, hue='Categoria', palette=custom_palette, 
                width=0.5, linewidth=2, fliersize=5, ax=ax3, legend=False)
    
    ax3.set_title('3. Analisi dei Quartili e Valori Anomali (Box Plot)', fontsize=12, pad=8)
    ax3.set_ylabel('Win Rate %', fontsize=10)
    ax3.set_xlabel('')

    ax4 = axs[1, 1]
    barplot = sns.barplot(x='Categoria', y='Partite', data=df, hue='Categoria', estimator=np.sum, 
                        palette=custom_palette, errorbar=None, ax=ax4, edgecolor="black", legend=False)
    
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
    plt.savefig('dashboard_analisi.png', dpi=150, bbox_inches='tight')
    print("[OK] Dashboard salvato come 'dashboard_analisi.png'")
    plt.close(fig)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=df, x='Costo_Elisir', y='Win_Rate',
                    hue='Categoria', palette=custom_palette,
                    s=80, alpha=0.6, edgecolor="black", linewidth=1, ax=ax1)
    sns.regplot(data=df, x='Costo_Elisir', y='Win_Rate', scatter=False,
                color='#2c3e50', line_kws={'linestyle': '--', 'linewidth': 2}, ax=ax1)
    ax1.set_title('Costo vs Win Rate', fontsize=13, pad=8)
    ax1.set_xlabel('Costo Elisir Medio', fontsize=10)
    ax1.set_ylabel('Win Rate %', fontsize=10)
    sns.despine()
    fig1.tight_layout()
    fig1.savefig('grafico_1_costo_vs_win_rate.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.kdeplot(data=df, x='Win_Rate', hue='Categoria', palette=custom_palette,
                fill=True, alpha=0.3, linewidth=3, common_norm=False, ax=ax2)
    ax2.set_title('Distribuzioni di Probabilità (Gaussiane)', fontsize=13, pad=8)
    ax2.set_xlabel('Win Rate %', fontsize=10)
    ax2.set_ylabel('Densità', fontsize=10)
    legend = ax2.get_legend()
    legend.set_title('Categoria')
    legend.set_bbox_to_anchor((0.98, 0.98))
    legend.set_loc('upper right')
    sns.despine()
    fig2.tight_layout()
    fig2.savefig('grafico_2_distribuzioni.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 7))
    sns.boxplot(x='Categoria', y='Win_Rate', data=df, hue='Categoria', palette=custom_palette,
                width=0.5, linewidth=2, fliersize=5, ax=ax3, legend=False)
    ax3.set_title('Analisi Quartili e Valori Anomali (Box Plot)', fontsize=13, pad=8)
    ax3.set_ylabel('Win Rate %', fontsize=10)
    ax3.set_xlabel('')
    sns.despine()
    fig3.tight_layout()
    fig3.savefig('grafico_3_box_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(9, 7))
    barplot4 = sns.barplot(x='Categoria', y='Partite', data=df, hue='Categoria', estimator=np.sum,
                           palette=custom_palette, errorbar=None, ax=ax4, edgecolor="black", legend=False)
    ax4.set_title('Volume Totale Partite Giocate', fontsize=13, pad=8)
    ax4.set_ylabel('Totale Partite', fontsize=10)
    ax4.set_xlabel('')
    for p in barplot4.patches:
        height = p.get_height()
        ax4.text(p.get_x() + p.get_width() / 2., height + 1000,
                 f'{int(height):,}', ha="center", fontsize=10, fontweight='bold', color='#2c3e50')
    sns.despine()
    fig4.tight_layout()
    fig4.savefig('grafico_4_volume_partite.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)

    print("[OK] Creati anche i PNG singoli:")
    print("     - grafico_1_costo_vs_win_rate.png")
    print("     - grafico_2_distribuzioni.png")
    print("     - grafico_3_box_plot.png")
    print("     - grafico_4_volume_partite.png")
    print("     - grafico_7_top5_combinato.png")
    # Uncomment the line below if running in an interactive environment with a display
    # plt.show()

def main():
    file_path = 'dataset_clash.csv'
    dataset = carica_dataset(file_path)
    dataset = valida_dataset(dataset)
    genera_report_testuale(dataset)
    calcola_bayes(dataset)
    mostra_grafici(dataset)
    grafico_top_5_combinato(dataset)

if __name__ == "__main__":
    main()