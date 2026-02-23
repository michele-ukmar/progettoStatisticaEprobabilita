import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE GLOBALE
# ──────────────────────────────────────────────────────────────────────────────

COLONNE_OBBLIGATORIE = {"Categoria", "Win_Rate", "Costo_Elisir", "Partite", "winCon"}
OUTPUT_DIR = Path("grafici")
OUTPUT_DIR.mkdir(exist_ok=True)

# Palette premium coerente in tutto il progetto
PALETTE = {
    "Cycle":    "#00C9FF",   # Ciano brillante
    "Beatdown": "#FF4C6B",   # Rosso corallo
    "Midrange": "#A8FF78",   # Verde lime
}

# Colori aggiuntivi
COL_GOLD   = "#FFD700"
COL_PURPLE = "#9B59B6"
COL_BG     = "#0F0F1A"      # Sfondo quasi nero
COL_PANEL  = "#1A1A2E"      # Pannello carta scura
COL_GRID   = "#2A2A45"      # Linee griglia sottili
COL_TEXT   = "#E8E8F0"      # Testo chiaro
COL_MUTED  = "#7070A0"      # Testo secondario


def _stile_dark(fig, axes):
    """Applica tema dark premium a figura e lista di assi."""
    fig.patch.set_facecolor(COL_BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(COL_PANEL)
        ax.tick_params(colors=COL_TEXT, labelsize=10)
        ax.xaxis.label.set_color(COL_TEXT)
        ax.yaxis.label.set_color(COL_TEXT)
        ax.title.set_color(COL_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(COL_GRID)
        ax.grid(color=COL_GRID, linestyle="--", linewidth=0.6, alpha=0.8)


def _titolo_grafico(ax, testo, sottotitolo=None):
    """Aggiunge titolo principale + eventuale sottotitolo stilizzati."""
    ax.set_title(testo, fontsize=15, fontweight="bold", color=COL_TEXT,
                 pad=18, loc="center")
    if sottotitolo:
        ax.annotate(sottotitolo, xy=(0.5, 1.01), xycoords="axes fraction",
                    ha="center", fontsize=9, color=COL_MUTED, style="italic")


def _salva(fig, nome_file):
    """Salva il grafico con padding e alta risoluzione."""
    percorso = OUTPUT_DIR / nome_file
    fig.savefig(percorso, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] Salvato: '{percorso}'")
    plt.close(fig)


def percorso_output(nome_file):
    return OUTPUT_DIR / nome_file


# ──────────────────────────────────────────────────────────────────────────────
# CARICAMENTO & VALIDAZIONE
# ──────────────────────────────────────────────────────────────────────────────

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
        raise ValueError("Il dataset è vuoto dopo la pulizia dei dati mancanti.")
    df = df[df['Categoria'] != 'Category']
    df['Win_Rate']     = pd.to_numeric(df['Win_Rate'],     errors='coerce')
    df['Costo_Elisir'] = pd.to_numeric(df['Costo_Elisir'], errors='coerce')
    df['Partite']      = pd.to_numeric(df['Partite'],      errors='coerce')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# REPORT TESTUALE
# ──────────────────────────────────────────────────────────────────────────────

def genera_report_testuale(df):
    cycle    = df[df['Categoria'] == 'Cycle']['Win_Rate']
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

    corr   = df['Costo_Elisir'].corr(df['Win_Rate'])
    validi = df[["Costo_Elisir", "Win_Rate"]].dropna()
    cov    = np.cov(validi["Costo_Elisir"], validi["Win_Rate"])[0][1] if not validi.empty else np.nan

    print(f"\n2. RELAZIONE COSTO-WIN RATE")
    print(f"   - Covarianza:   {cov:.4f}")
    print(f"   - Correlazione: {corr:.4f}")
    if abs(corr) < 0.3:
        print("   -> INTERPRETAZIONE: Non esiste una correlazione significativa.")
        print("      Vincere non dipende dal costo del mazzo (Skill > Elisir).")


def calcola_bayes(df, soglia_top=52.0):
    top_tier         = df[df['Win_Rate'] > soglia_top]
    prob_top_tier    = len(top_tier) / len(df)
    beatdown_top     = top_tier[top_tier['Categoria'] == 'Beatdown']
    prob_intersezione = len(beatdown_top) / len(df)
    if prob_top_tier > 0:
        prob_condizionata = prob_intersezione / prob_top_tier
        print(f"\n3. PROBABILITÀ CONDIZIONATA (BAYES)")
        print(f"   - Probabilità che un mazzo di fascia alta sia Beatdown: {prob_condizionata*100:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICO 1 — Scatter Costo Elisir vs Win Rate
# ──────────────────────────────────────────────────────────────────────────────

def grafico_costo_vs_winrate(df):
    fig, ax = plt.subplots(figsize=(12, 7))
    _stile_dark(fig, ax)

    for cat, col in PALETTE.items():
        sub = df[df['Categoria'] == cat]
        ax.scatter(sub['Costo_Elisir'], sub['Win_Rate'],
                   c=col, s=65, alpha=0.82, edgecolors="white",
                   linewidths=0.4, label=cat, zorder=3)

    # Linea di tendenza
    validi = df[["Costo_Elisir", "Win_Rate"]].dropna()
    m, b, *_ = stats.linregress(validi["Costo_Elisir"], validi["Win_Rate"])
    xs = np.linspace(validi["Costo_Elisir"].min(), validi["Costo_Elisir"].max(), 200)
    ax.plot(xs, m * xs + b, color=COL_GOLD, linewidth=2, linestyle="--",
            alpha=0.8, label=f"Trend (r={validi['Costo_Elisir'].corr(validi['Win_Rate']):.2f})", zorder=4)

    # Zona 50% win rate
    ax.axhline(50, color=COL_MUTED, linewidth=1.2, linestyle=":", alpha=0.7, zorder=2)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else validi["Costo_Elisir"].min(),
            50.3, "  Win Rate = 50%", color=COL_MUTED, fontsize=8, va="bottom")

    ax.set_xlabel("Costo Medio Elisir", fontsize=12, labelpad=10)
    ax.set_ylabel("Win Rate (%)", fontsize=12, labelpad=10)
    _titolo_grafico(ax, "Costo Elisir vs Win Rate per Categoria",
                    "Ogni punto = 1 mazzo analizzato · 500 mazzi totali")

    leg = ax.legend(frameon=True, facecolor=COL_BG, edgecolor=COL_GRID,
                    labelcolor=COL_TEXT, fontsize=10, loc="upper right")
    ax.tick_params(axis='both', which='both', length=4)

    _salva(fig, "grafico_1_costo_vs_win_rate.png")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICO 2 — Distribuzioni KDE Win Rate per categoria
# ──────────────────────────────────────────────────────────────────────────────

def grafico_distribuzioni(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    _stile_dark(fig, axes)
    fig.subplots_adjust(wspace=0.35, top=0.88)

    categorie = ["Cycle", "Beatdown", "Midrange"]
    
    # Calcola il range X globale per uniformità
    all_wr = df[df['Categoria'].isin(categorie)]['Win_Rate'].dropna()
    x_min = all_wr.min() - 1
    x_max = all_wr.max() + 1
    x_range = np.linspace(x_min, x_max, 300)
    
    # Calcola i valori Y massimi per trovare il limite Y comune
    y_max_global = 0
    distributions = {}
    for cat in categorie:
        sub = df[df['Categoria'] == cat]['Win_Rate'].dropna()
        kde = stats.gaussian_kde(sub)
        kde_y = kde(x_range)
        distributions[cat] = (sub, kde, kde_y)
        y_max_global = max(y_max_global, kde_y.max())
    
    for ax, cat in zip(axes, categorie):
        sub, kde, kde_y = distributions[cat]
        col  = PALETTE[cat]

        # KDE riempito con X uniforme
        ax.fill_between(x_range, kde_y, alpha=0.25, color=col)
        ax.plot(x_range, kde_y, color=col, linewidth=2.5)

        # Media e mediana
        ax.axvline(sub.mean(),   color=COL_GOLD,   linestyle="--", linewidth=1.5,
                   label=f"Media {sub.mean():.1f}%")
        ax.axvline(sub.median(), color="white",     linestyle=":",  linewidth=1.2,
                   label=f"Mediana {sub.median():.1f}%")

        # Annotazione std
        ax.text(0.97, 0.94, f"σ = {sub.std():.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COL_BG,
                          edgecolor=col, linewidth=1.2))

        # Imposta limiti uguali per tutti i subplot
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max_global * 1.05)
        
        _titolo_grafico(ax, cat)
        ax.set_xlabel("Win Rate (%)", fontsize=10)
        ax.set_ylabel("Densità", fontsize=10)
        leg = ax.legend(fontsize=8, frameon=True, facecolor=COL_BG,
                        edgecolor=COL_GRID, labelcolor=COL_TEXT)

    fig.suptitle("Distribuzione Win Rate per Categoria",
                 fontsize=16, fontweight="bold", color=COL_TEXT, y=1.01)
    _salva(fig, "grafico_2_distribuzioni.png")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICO 3 — Box Plot Win Rate per categoria
# ──────────────────────────────────────────────────────────────────────────────

def grafico_box_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    _stile_dark(fig, ax)

    categorie = ["Cycle", "Beatdown", "Midrange"]
    dati      = [df[df['Categoria'] == c]['Win_Rate'].dropna().values for c in categorie]
    colori    = [PALETTE[c] for c in categorie]

    bp = ax.boxplot(dati, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, col in zip(bp['boxes'], colori):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_color(COL_MUTED)
    for cap in bp['caps']:
        cap.set_color(COL_MUTED)
    for flier in bp['fliers']:
        flier.set_markerfacecolor(COL_MUTED)
        flier.set_markeredgecolor(COL_MUTED)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(categorie, fontsize=12, fontweight="bold")
    for tick, col in zip(ax.get_xticklabels(), colori):
        tick.set_color(col)

    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.axhline(50, color=COL_MUTED, linewidth=1, linestyle=":", alpha=0.6)
    _titolo_grafico(ax, "Distribuzione Win Rate per Categoria (Box Plot)",
                    "Confronto variabilità e mediana tra i tre stili di gioco")
    _salva(fig, "grafico_3_box_plot.png")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICO 4 — Volume Partite per categoria (Bar orizzontale)
# ──────────────────────────────────────────────────────────────────────────────

def grafico_volume_partite(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    _stile_dark(fig, ax)

    totali = df.groupby('Categoria')['Partite'].sum().sort_values(ascending=True)
    categorie = totali.index.tolist()
    valori    = totali.values
    colori    = [PALETTE.get(c, "#888888") for c in categorie]

    bars = ax.barh(categorie, valori, color=colori, alpha=0.85,
                   edgecolor="white", linewidth=0.8, height=0.55)

    for bar, val, col in zip(bars, valori, colori):
        ax.text(val + totali.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=col)

    for label, col in zip(ax.get_yticklabels(), colori):
        label.set_color(col)
        label.set_fontweight("bold")

    ax.set_xlabel("Partite Totali Giocate", fontsize=11)
    ax.set_xlim(0, totali.max() * 1.16)
    ax.grid(axis='x', color=COL_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(axis='y', visible=False)
    _titolo_grafico(ax, "Volume Totale Partite per Categoria",
                    "Somma di tutte le partite giocate dai mazzi di ogni stile")
    _salva(fig, "grafico_4_volume_partite.png")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICO 5 — Win Condition Pie (donut)
# ──────────────────────────────────────────────────────────────────────────────

def grafico_wincondition_pie(df):
    conteggi  = df['winCon'].value_counts()
    top_n     = 8
    top_items = conteggi.head(top_n)
    altri     = conteggi.iloc[top_n:].sum()
    if altri > 0:
        top_items = pd.concat([top_items, pd.Series({"Altre": altri})])

    labels  = top_items.index.tolist()
    sizes   = top_items.values

    # Palette vibrante per le fette
    cmap   = plt.cm.get_cmap("plasma", len(labels))
    colori = [cmap(i) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11, 8))
    _stile_dark(fig, ax)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%',
        colors=colori,
        pctdistance=0.78,
        startangle=140,
        wedgeprops=dict(width=0.55, edgecolor=COL_BG, linewidth=2.5),
        textprops=dict(color=COL_TEXT, fontsize=9)
    )

    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
        at.set_color("white")
        at.set_path_effects([pe.withStroke(linewidth=2, foreground=COL_BG)])

    # Cerchio centrale (testo totale)
    centre_circle = plt.Circle((0, 0), 0.42, fc=COL_PANEL, linewidth=2,
                                edgecolor=COL_GRID)
    ax.add_artist(centre_circle)
    ax.text(0, 0.06, "Win\nConditions", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COL_TEXT)
    ax.text(0, -0.14, f"{len(df['winCon'].unique())} tipologie",
            ha="center", va="center", fontsize=8, color=COL_MUTED)

    # Legenda laterale
    legend_patches = [mpatches.Patch(facecolor=c, label=l, edgecolor="white",
                                     linewidth=0.5)
                      for c, l in zip(colori, labels)]
    ax.legend(handles=legend_patches, loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=9,
              frameon=True, facecolor=COL_BG, edgecolor=COL_GRID,
              labelcolor=COL_TEXT, handlelength=1.2)

    ax.set_title("Distribuzione Win Conditions",
                 fontsize=15, fontweight="bold", color=COL_TEXT, pad=20)
    ax.annotate("Top 8 win condition + Altre",
                xy=(0.5, -0.02), xycoords="axes fraction",
                ha="center", fontsize=8, color=COL_MUTED, style="italic")

    _salva(fig, "grafico_wincondition_pie.png")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICO 6 — Heatmap Media Win Rate per categoria x fascia costo
# ──────────────────────────────────────────────────────────────────────────────

def grafico_heatmap(df):
    df2      = df.copy()
    bins     = [0, 2.5, 3.0, 3.5, 4.0, 4.5, 10]
    labels_b = ["≤2.5", "2.6-3.0", "3.1-3.5", "3.6-4.0", "4.1-4.5", "≥4.6"]
    df2['Fascia_Costo'] = pd.cut(df2['Costo_Elisir'], bins=bins, labels=labels_b)

    pivot = df2.pivot_table(values='Win_Rate', index='Categoria',
                            columns='Fascia_Costo', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(12, 5))
    _stile_dark(fig, ax)

    cmap_custom = LinearSegmentedColormap.from_list(
        "cr_heat", ["#0F0F1A", "#9B59B6", "#FF4C6B", "#FFD700"], N=256)

    sns.heatmap(pivot, ax=ax, cmap=cmap_custom, annot=True, fmt=".1f",
                linewidths=1.5, linecolor=COL_BG,
                cbar_kws={"label": "Win Rate Medio (%)", "shrink": 0.8},
                annot_kws={"fontsize": 10, "fontweight": "bold", "color": "white"})

    ax.set_xlabel("Fascia Costo Elisir", fontsize=11, color=COL_TEXT, labelpad=10)
    ax.set_ylabel("Categoria", fontsize=11, color=COL_TEXT, labelpad=10)
    ax.tick_params(colors=COL_TEXT, labelsize=10)

    for y, cat in enumerate(pivot.index):
        col = PALETTE.get(cat, "white")
        ax.get_yticklabels()[y].set_color(col)
        ax.get_yticklabels()[y].set_fontweight("bold")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(COL_TEXT)
    cbar.ax.tick_params(colors=COL_TEXT)

    ax.set_title("Win Rate Medio per Categoria e Fascia di Costo",
                 fontsize=14, fontweight="bold", color=COL_TEXT, pad=16)
    _salva(fig, "grafico_5_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICO 7 — Top 5 combinato (Win Rate + Partite)
# ──────────────────────────────────────────────────────────────────────────────

def grafico_top_5_combinato(df):
    df_top5_wr = df.nlargest(5, 'Win_Rate')[['Mazzo', 'Win_Rate', 'Partite', 'Categoria']].reset_index(drop=True)
    df_top5_pt = df.nlargest(5, 'Partite')[['Mazzo', 'Win_Rate', 'Partite', 'Categoria']].reset_index(drop=True)
    df_top5_wr['Gruppo'] = 'Più Vincenti'
    df_top5_pt['Gruppo'] = 'Più Giocati'
    df_combined = pd.concat([df_top5_wr, df_top5_pt], ignore_index=True)

    max_wr = max(df_top5_wr['Win_Rate'].max(), df_top5_pt['Win_Rate'].max())
    max_pt = max(df_top5_wr['Partite'].max(), df_top5_pt['Partite'].max())

    fig, ax_main = plt.subplots(figsize=(18, 8))
    _stile_dark(fig, ax_main)

    colors = [PALETTE.get(cat, "#888888") for cat in df_combined['Categoria']]
    x      = np.arange(len(df_combined))
    width  = 0.36

    bars_wr = ax_main.bar(x - width / 2, df_combined['Win_Rate'], width,
                          color=colors, alpha=0.88, edgecolor="white",
                          linewidth=1.2, zorder=3)

    ax_aux = ax_main.twinx()
    ax_aux.set_facecolor('none')
    bars_pt = ax_aux.bar(x + width / 2, df_combined['Partite'], width,
                         color=COL_GOLD, alpha=0.72, edgecolor="white",
                         linewidth=1.2, zorder=3)

    # Assi
    ax_main.set_ylabel("Win Rate %", fontsize=12, color=COL_TEXT, labelpad=10)
    ax_main.set_ylim(44, max_wr + 3)
    ax_main.tick_params(axis='y', colors=COL_TEXT, labelsize=10)
    ax_main.tick_params(axis='x', colors=COL_TEXT, labelsize=9)
    ax_main.grid(axis='y', color=COL_GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax_main.grid(axis='x', visible=False)

    ax_aux.set_ylabel("Partite Giocate", fontsize=12, color=COL_GOLD, labelpad=10)
    ax_aux.set_ylim(0, max_pt * 1.18)
    ax_aux.tick_params(axis='y', colors=COL_GOLD, labelsize=10)
    for spine in ax_aux.spines.values():
        spine.set_edgecolor(COL_GRID)
    ax_aux.grid(visible=False)

    # Labels x
    ax_main.set_xticks(x)
    nomi = [m[:22] for m in df_combined['Mazzo']]
    ax_main.set_xticklabels(nomi, rotation=38, ha='right', fontsize=9, color=COL_TEXT)

    # Separatore gruppi
    ax_main.axvline(x=4.5, color=COL_MUTED, linestyle="--", linewidth=2, alpha=0.5, zorder=2)

    # Etichette sezioni
    ypos = max_wr + 2.2
    for xc, label in [(2, "● Mazzi Più Vincenti"), (7, "● Mazzi Più Giocati")]:
        ax_main.text(xc, ypos, label, fontsize=10, fontweight="bold",
                     color=COL_TEXT, ha="center",
                     bbox=dict(boxstyle='round,pad=0.4', facecolor=COL_PANEL,
                               edgecolor=COL_GRID, linewidth=1.2))

    # Valori sulle barre
    for bar, val, col in zip(bars_wr, df_combined['Win_Rate'], colors):
        ax_main.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
                     fontweight='bold', color=col)

    for bar, val in zip(bars_pt, df_combined['Partite']):
        ax_aux.text(bar.get_x() + bar.get_width() / 2, val + max_pt * 0.008,
                    f'{int(val):,}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color=COL_GOLD)

    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["Beatdown"], edgecolor="white", label="Beatdown"),
        mpatches.Patch(facecolor=PALETTE["Cycle"],    edgecolor="white", label="Cycle"),
        mpatches.Patch(facecolor=PALETTE["Midrange"], edgecolor="white", label="Midrange"),
        mpatches.Patch(facecolor=COL_GOLD,            edgecolor="white", label="Partite Giocate"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9, facecolor=COL_BG,
               edgecolor=COL_GRID, labelcolor=COL_TEXT,
               bbox_to_anchor=(0.5, -0.06))

    ax_main.set_title("Top 5 Mazzi Più Vincenti vs Più Giocati",
                      fontsize=15, fontweight="bold", color=COL_TEXT, pad=18)
    plt.subplots_adjust(bottom=0.22)
    _salva(fig, "grafico_7_top5_combinato.png")


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD — tutti i pannelli in una sola immagine
# ──────────────────────────────────────────────────────────────────────────────

def genera_dashboard(df):
    fig = plt.figure(figsize=(22, 20), facecolor=COL_BG)
    fig.suptitle("Clash Royale — Analisi Statistica Mazzi\n500 mazzi · 3 categorie · Dataset 2024",
                 fontsize=18, fontweight="bold", color=COL_TEXT, y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    # ── pannello 1: scatter costo vs win rate ──
    ax1 = fig.add_subplot(gs[0, 0:2])
    _stile_dark(fig, ax1)
    for cat, col in PALETTE.items():
        sub = df[df['Categoria'] == cat]
        ax1.scatter(sub['Costo_Elisir'], sub['Win_Rate'],
                    c=col, s=40, alpha=0.75, edgecolors="white",
                    linewidths=0.3, label=cat, zorder=3)
    validi = df[["Costo_Elisir", "Win_Rate"]].dropna()
    m, b, *_ = stats.linregress(validi["Costo_Elisir"], validi["Win_Rate"])
    xs = np.linspace(validi["Costo_Elisir"].min(), validi["Costo_Elisir"].max(), 200)
    ax1.plot(xs, m * xs + b, color=COL_GOLD, linewidth=1.8, linestyle="--", alpha=0.8, zorder=4)
    ax1.axhline(50, color=COL_MUTED, linewidth=1, linestyle=":", alpha=0.6)
    ax1.set_xlabel("Costo Elisir", fontsize=10)
    ax1.set_ylabel("Win Rate (%)", fontsize=10)
    ax1.legend(fontsize=8, frameon=True, facecolor=COL_BG, edgecolor=COL_GRID, labelcolor=COL_TEXT)
    ax1.set_title("Costo Elisir vs Win Rate", fontsize=11, fontweight="bold", color=COL_TEXT)

    # ── pannello 2: box plot ──
    ax2 = fig.add_subplot(gs[0, 2])
    _stile_dark(fig, ax2)
    categorie = ["Cycle", "Beatdown", "Midrange"]
    dati = [df[df['Categoria'] == c]['Win_Rate'].dropna().values for c in categorie]
    bp = ax2.boxplot(dati, patch_artist=True, notch=True,
                     medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(linewidth=1.2, color=COL_MUTED),
                     capprops=dict(linewidth=1.2, color=COL_MUTED),
                     flierprops=dict(marker='o', markersize=3, alpha=0.4,
                                     markerfacecolor=COL_MUTED, markeredgecolor=COL_MUTED))
    for patch, cat in zip(bp['boxes'], categorie):
        patch.set_facecolor(PALETTE[cat]); patch.set_alpha(0.65); patch.set_edgecolor("white")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(categorie, fontsize=9)
    for t, cat in zip(ax2.get_xticklabels(), categorie):
        t.set_color(PALETTE[cat]); t.set_fontweight("bold")
    ax2.axhline(50, color=COL_MUTED, linewidth=1, linestyle=":", alpha=0.6)
    ax2.set_title("Box Plot Win Rate", fontsize=11, fontweight="bold", color=COL_TEXT)

    # ── pannello 3-4: KDE distribuzione ──
    for idx, cat in enumerate(categorie):
        ax = fig.add_subplot(gs[1, idx])
        _stile_dark(fig, ax)
        sub = df[df['Categoria'] == cat]['Win_Rate'].dropna()
        col = PALETTE[cat]
        kde_x = np.linspace(sub.min() - 0.5, sub.max() + 0.5, 300)
        kde_y = stats.gaussian_kde(sub)(kde_x)
        ax.fill_between(kde_x, kde_y, alpha=0.22, color=col)
        ax.plot(kde_x, kde_y, color=col, linewidth=2.2)
        ax.axvline(sub.mean(), color=COL_GOLD, linestyle="--", linewidth=1.4,
                   label=f"μ={sub.mean():.1f}%")
        ax.text(0.96, 0.93, f"σ={sub.std():.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=col,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=COL_BG,
                          edgecolor=col, linewidth=1))
        ax.legend(fontsize=7, frameon=True, facecolor=COL_BG,
                  edgecolor=COL_GRID, labelcolor=COL_TEXT)
        ax.set_xlabel("Win Rate (%)", fontsize=9)
        ax.set_ylabel("Densità", fontsize=9)
        ax.set_title(f"KDE — {cat}", fontsize=10, fontweight="bold", color=col)

    # ── pannello 5: volume partite ──
    ax5 = fig.add_subplot(gs[2, 0])
    _stile_dark(fig, ax5)
    totali  = df.groupby('Categoria')['Partite'].sum().sort_values(ascending=True)
    colori5 = [PALETTE.get(c, "#888") for c in totali.index]
    bars5   = ax5.barh(totali.index, totali.values, color=colori5, alpha=0.85,
                       edgecolor="white", linewidth=0.7, height=0.5)
    for bar, val, col in zip(bars5, totali.values, colori5):
        ax5.text(val + totali.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{int(val):,}", va="center", ha="left", fontsize=9,
                 fontweight="bold", color=col)
    for t, col in zip(ax5.get_yticklabels(), reversed(colori5)):
        t.set_color(col); t.set_fontweight("bold")
    ax5.set_xlabel("Partite Totali", fontsize=9)
    ax5.set_xlim(0, totali.max() * 1.22)
    ax5.grid(axis='x', color=COL_GRID, linestyle="--", linewidth=0.5, alpha=0.7)
    ax5.grid(axis='y', visible=False)
    ax5.set_title("Volume Partite per Categoria", fontsize=10, fontweight="bold", color=COL_TEXT)

    # ── pannello 6: win condition donut ──
    ax6 = fig.add_subplot(gs[2, 1])
    _stile_dark(fig, ax6)
    conteggi = df['winCon'].value_counts()
    top_n    = 6
    top_it   = conteggi.head(top_n)
    altri    = conteggi.iloc[top_n:].sum()
    if altri > 0:
        top_it = pd.concat([top_it, pd.Series({"Altre": altri})])
    cmap6  = plt.cm.get_cmap("plasma", len(top_it))
    col6   = [cmap6(i) for i in range(len(top_it))]
    ax6.pie(top_it.values, labels=None, colors=col6,
            startangle=140, pctdistance=0.78,
            autopct='%1.0f%%',
            wedgeprops=dict(width=0.5, edgecolor=COL_BG, linewidth=2),
            textprops=dict(color="white", fontsize=7))
    cc = plt.Circle((0, 0), 0.45, fc=COL_PANEL, linewidth=1.5, edgecolor=COL_GRID)
    ax6.add_artist(cc)
    ax6.text(0, 0, "Win\nCon", ha="center", va="center",
             fontsize=9, fontweight="bold", color=COL_TEXT)
    legend_p = [mpatches.Patch(facecolor=c, label=l[:14], edgecolor="white", linewidth=0.5)
                for c, l in zip(col6, top_it.index)]
    ax6.legend(handles=legend_p, fontsize=6.5, frameon=True, facecolor=COL_BG,
               edgecolor=COL_GRID, labelcolor=COL_TEXT,
               loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax6.set_title("Win Conditions", fontsize=10, fontweight="bold", color=COL_TEXT)

    # ── pannello 7: heatmap ──
    ax7 = fig.add_subplot(gs[2, 2])
    _stile_dark(fig, ax7)
    df2 = df.copy()
    bins2   = [0, 2.5, 3.0, 3.5, 4.0, 4.5, 10]
    labels2 = ["≤2.5", "2.6-3.0", "3.1-3.5", "3.6-4.0", "4.1-4.5", "≥4.6"]
    df2['Fascia_Costo'] = pd.cut(df2['Costo_Elisir'], bins=bins2, labels=labels2)
    pivot2 = df2.pivot_table(values='Win_Rate', index='Categoria',
                             columns='Fascia_Costo', aggfunc='mean')
    cmap7 = LinearSegmentedColormap.from_list(
        "cr_heat", ["#0F0F1A", "#9B59B6", "#FF4C6B", "#FFD700"], N=256)
    sns.heatmap(pivot2, ax=ax7, cmap=cmap7, annot=True, fmt=".1f",
                linewidths=1.2, linecolor=COL_BG,
                cbar_kws={"shrink": 0.7},
                annot_kws={"fontsize": 7.5, "fontweight": "bold", "color": "white"})
    ax7.set_xlabel("Fascia Costo", fontsize=8, color=COL_TEXT)
    ax7.set_ylabel("Categoria", fontsize=8, color=COL_TEXT)
    ax7.tick_params(colors=COL_TEXT, labelsize=7)
    for t, cat in zip(ax7.get_yticklabels(), pivot2.index):
        t.set_color(PALETTE.get(cat, "white"))
        t.set_fontweight("bold")
    cbar7 = ax7.collections[0].colorbar
    cbar7.ax.tick_params(colors=COL_TEXT, labelsize=7)
    ax7.set_title("Heatmap Win Rate", fontsize=10, fontweight="bold", color=COL_TEXT)

    _salva(fig, "dashboard_analisi.png")


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATORE GRAFICI SINGOLI
# ──────────────────────────────────────────────────────────────────────────────

def mostra_grafici(df):
    print("\n[INFO] Generazione grafici singoli...")
    grafico_costo_vs_winrate(df)
    grafico_distribuzioni(df)
    grafico_box_plot(df)
    grafico_volume_partite(df)
    grafico_heatmap(df)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    file_path = 'dataset_clash.csv'
    dataset   = carica_dataset(file_path)
    dataset   = valida_dataset(dataset)
    genera_report_testuale(dataset)
    calcola_bayes(dataset)
    mostra_grafici(dataset)
    grafico_top_5_combinato(dataset)
    grafico_wincondition_pie(dataset)
    genera_dashboard(dataset)


if __name__ == "__main__":
    main()
