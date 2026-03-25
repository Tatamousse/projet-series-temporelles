import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from base64 import b64encode
import os
from flask import Flask, render_template, request
from lib_2026 import *

import math
import numpy
import pandas
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# CHARGEMENT DES DONNÉES

fichier = "./static/data/regularite-mensuelle-ter.csv"
df_TER = pandas.read_csv(fichier, sep=";")

# Ordonner les données par Région et année-mois
df_TER = df_TER.sort_values(by=[df_TER.columns[1], df_TER.columns[0]])

dic_Region = dict(df_TER[df_TER.columns[1]].value_counts())

# Régions avec au moins 40 données
Liste_Region = sorted([k for k, v in dic_Region.items() if v > 40])

# APPLICATION FLASK

app = Flask(__name__)


@app.route('/')
@app.route("/index")
def index():
    return render_template('index.html', Regions=Liste_Region)


@app.route("/Resultats", methods=['GET', 'POST'])
def Resultats():
    if request.method == 'POST':
        choix = str(request.form['Choix_region'])

        # Extraction des données pour la région choisie
        df_Choix = df_TER[df_TER[df_TER.columns[1]] == choix][
            [df_TER.columns[0], df_TER.columns[4]]
        ].dropna()

        Date_Choix = df_Choix[df_Choix.columns[0]].values.tolist()
        Trains_Annul_Choix = df_Choix[df_Choix.columns[1]].values.tolist()
        X = [int(u) for u in Trains_Annul_Choix]

        # Graphe de la série temporelle
        plot_html = graphe_serie64(X, Date_Choix, choix)

        # Statistiques descriptives
        Stats = stats_descriptives(X)

        # Histogramme
        histo_html = Histo_Continue64(X, 9)

        # Détection de rupture Skew Normal
        k_hat, params1, params2, log_vraisemblances = detection_rupture_SN(X)

        estimation_resultats = {}
        rupture_html = ""
        logvrai_html = ""
        densite_html = ""
        ks_resultats = {}

        if k_hat is not None and params1 is not None and params2 is not None:
            # Paramètres estimés
            estimation_resultats = {
                "Point de rupture (k̂)": k_hat,
                "Date de rupture": Date_Choix[k_hat] if k_hat < len(Date_Choix) else "N/A",
                "μ̂₁ (avant rupture)": float("{:.4f}".format(params1[0])),
                "σ̂₁ (avant rupture)": float("{:.4f}".format(params1[1])),
                "μ̂₂ (après rupture)": float("{:.4f}".format(params2[0])),
                "σ̂₂ (après rupture)": float("{:.4f}".format(params2[1])),
                "θ̂ (asymétrie)": float("{:.4f}".format(params1[2])),
            }

            # Test de Kolmogorov-Smirnov
            ks_stat, ks_pvalue = test_KS(X, k_hat)
            alpha = 0.05
            decision = "Rejet de H₀ : il y a une rupture" if ks_pvalue < alpha else "Non-rejet de H₀ : pas de rupture significative"
            ks_resultats = {
                "Statistique KS": float("{:.4f}".format(ks_stat)),
                "p-value": float("{:.6f}".format(ks_pvalue)),
                "Seuil α": alpha,
                "Décision": decision,
            }

            # Graphiques
            rupture_html = graphe_rupture64(X, Date_Choix, k_hat, params1, params2, choix)
            logvrai_html = graphe_log_vraisemblance64(log_vraisemblances, k_hat, choix)
            densite_html = graphe_densite_ajustee64(X, k_hat, params1, params2, choix)

        return render_template("Resultat_TER.html",
                               choix=choix,
                               Regions=Liste_Region,
                               Stats_html=Stats,
                               plot_html=plot_html,
                               histo_html=histo_html,
                               estimation_resultats=estimation_resultats,
                               ks_resultats=ks_resultats,
                               rupture_html=rupture_html,
                               logvrai_html=logvrai_html,
                               densite_html=densite_html,
                               )
    else:
        return render_template('index.html', Regions=Liste_Region)


if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:8080/")
    app.run(debug=True, port=8080, use_reloader=False, threaded=True)
