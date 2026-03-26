import numpy
import math
import functools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from base64 import b64encode
from io import BytesIO
from scipy.stats import norm, ks_2samp
import scipy.optimize

# CHARGEMENT DES DONNÉES

def charger_donnees(chemin_fichier):
    """Charge le CSV, trie par région/date, retourne le DataFrame et la liste des régions valides."""
    import pandas
    df = pandas.read_csv(chemin_fichier, sep=";")
    df = df.sort_values(by=[df.columns[1], df.columns[0]])
    dic_Region = dict(df[df.columns[1]].value_counts())
    liste_region = sorted([k for k, v in dic_Region.items() if v > 40])
    return df, liste_region

# MOMENTS

def Moments_r(data, r):
    data = [x for x in data]
    fonc_r = lambda x: x**r
    S = functools.reduce(lambda x, y: x + y, map(fonc_r, data))
    return S / (1.0 * len(data))


def Moments_Centre_r(data, r):
    data = [x for x in data]
    m = Moments_r(data, 1)
    fonc_r = lambda x: (x - m)**r
    S = functools.reduce(lambda x, y: x + y, map(fonc_r, data))
    return S / (1.0 * len(data))


# STATISTIQUES DESCRIPTIVES

def stats_descriptives(data):
    data = [x for x in data]
    Resultats = {}
    n = len(data)
    m0 = min(data)
    m1 = max(data)
    m = Moments_r(data, 1)
    sigma2 = Moments_Centre_r(data, 2)
    sigma = math.sqrt(sigma2)

    gamma1 = Moments_Centre_r(data, 3) / sigma**3
    gamma2 = -3.0 + Moments_Centre_r(data, 4) / sigma**4

    Resultats["Nombre d'observations"] = int("{:d}".format(n))
    if int(m0) == float(m0):
        Resultats["minimum"] = int("{:d}".format(int(m0)))
    else:
        Resultats["minimum"] = float("{:.2f}".format(m0))
    if int(m1) == float(m1):
        Resultats["maximum"] = int("{:d}".format(int(m1)))
    else:
        Resultats["maximum"] = float("{:.2f}".format(m1))

    Resultats["moyenne empirique"] = float("{:.2f}".format(m))
    Resultats["variance empirique"] = float("{:.2f}".format(sigma2))
    Resultats["skewness"] = float("{:.4f}".format(gamma1))
    Resultats["kurtosis"] = float("{:.4f}".format(gamma2))

    return Resultats


# HISTOGRAMME EN BASE64

def Histo_Continue64(data, k):
    plt.rcParams['hatch.color'] = [0.9, 0.9, 0.9]
    data = numpy.array([x for x in data])
    Ext = [min(data) + (max(data) - min(data)) * i / (1.0 * k) for i in range(k + 1)]
    C = [0.5 * (Ext[i] + Ext[i + 1]) for i in range(k)]

    NN = []
    for i in range(k):
        NN.append(((Ext[i] <= data) & (data <= Ext[i + 1])).sum())

    indice_max = [i for i in range(k) if NN[i] == numpy.max(NN)]
    TT = [str("{:.1f}".format(t)) for t in Ext]

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
    largeur = Ext[1] - Ext[0]

    for i in range(k):
        if i in indice_max:
            ax1.bar(C[i], NN[i], largeur, color=[0.15, 0.15, 0.80],
                    edgecolor="white", hatch="/", lw=1., zorder=0, alpha=0.9)
        else:
            ax1.bar(C[i], NN[i], largeur, align='center', edgecolor="white")
        ax1.text(C[i], NN[i], "%d" % (NN[i]), fontsize=9, style='italic',
                 horizontalalignment='center', verticalalignment='bottom')

    ax1.axvline(x=numpy.mean(data), color='red', lw=4, label='valeur moyenne')
    ax1.set_xticks(Ext)
    ax1.set_xticklabels(TT, fontsize=9, rotation=45)
    ax1.set_xlim(numpy.min(data) - 0.75 * largeur, numpy.max(data) + 0.75 * largeur)
    ax1.set_ylim(0.0, numpy.max(NN) + 3.0)
    ax1.set_xlabel("Valeurs", fontsize=13, labelpad=0)
    ax1.set_ylabel("Effectifs", fontsize=14)
    plt.legend(loc='best')

    plt.savefig(buf := BytesIO(), format='png', bbox_inches='tight')
    plt.close()
    return b64encode(buf.getvalue()).decode()


# DENSITÉ SKEW NORMAL

def densite_skew_normal(x, mu, sigma, theta):
    """Densité de la loi Skew Normal SN(mu, sigma, theta)"""
    z = (x - mu) / sigma
    return (2.0 / sigma) * norm.pdf(z) * norm.cdf(theta * z)


# ESTIMATION MME (Méthode des Moments)

def estimation_MME(data):
    """
    Estimation des paramètres (xi, tau, theta) de SN(xi, tau, theta)
    par la méthode des moments (MME) - Ghorbanzadeh et al. WCE2017
    """
    data = numpy.array(data, dtype=float)
    n = len(data)
    xn = numpy.mean(data)

    # Moments centrés
    m2 = numpy.mean((data - xn)**2)
    m3 = numpy.mean((data - xn)**3)

    # Skewness
    gamma1 = m3 / (m2**1.5) if m2 > 0 else 0.0

    a = math.sqrt(2.0 / math.pi)
    b = (4.0 - math.pi) / 2.0

    # Equation (11) du paper
    abs_g = abs(gamma1)
    if abs_g < 1e-10:
        delta_tilde = 0.0
    else:
        abs_g_23 = abs_g**(2.0/3.0)
        b_23 = b**(2.0/3.0)
        delta_tilde = numpy.sign(gamma1) * math.sqrt(abs_g_23 / (a**2 * (b_23 + abs_g_23)))

    # Borner delta pour éviter les problèmes numériques
    delta_tilde = max(-0.9999, min(0.9999, delta_tilde))

    # Equation (12) du paper
    theta_tilde = delta_tilde / math.sqrt(1.0 - delta_tilde**2)
    tau_tilde = math.sqrt(m2 / (1.0 - a**2 * delta_tilde**2))
    xi_tilde = xn - a * tau_tilde * delta_tilde

    return xi_tilde, tau_tilde, theta_tilde


# ESTIMATION MLE (Maximum de Vraisemblance) 

def estimation_MLE(data):
    """
    Estimation des paramètres (xi, tau, theta) de SN(xi, tau, theta)
    par le maximum de vraisemblance (MLE) - Ghorbanzadeh et al. WCE2017
    """
    data = numpy.array(data, dtype=float)

    # Initialisation avec MME
    xi0, tau0, theta0 = estimation_MME(data)
    if tau0 <= 0:
        tau0 = numpy.std(data) + 1e-6
    # Borner theta initial
    theta0 = max(-50, min(50, theta0))

    def neg_log_likelihood(params):
        xi, tau, theta = params
        if tau <= 0:
            return 1e15
        # Borner theta pour la stabilité
        theta = max(-50, min(50, theta))
        z = (data - xi) / tau
        ll = numpy.sum(numpy.log(2.0 / tau + 1e-300) +
                       norm.logpdf(z) +
                       numpy.log(norm.cdf(theta * z) + 1e-300))
        return -ll

    try:
        res = scipy.optimize.minimize(neg_log_likelihood, [xi0, tau0, theta0],
                                       method='Nelder-Mead',
                                       options={'maxiter': 5000, 'disp': False})
        if not res.success or not numpy.isfinite(res.fun):
            return xi0, tau0, theta0
        xi_hat, tau_hat, theta_hat = res.x
        if tau_hat <= 0 or not numpy.isfinite(tau_hat):
            return xi0, tau0, theta0
        theta_hat = max(-50, min(50, theta_hat))
        return xi_hat, abs(tau_hat), theta_hat
    except Exception:
        return xi0, tau0, theta0


# MODÈLE DE DÉTECTION DE RUPTURE - LOI SKEW NORMAL

def log_vraisemblance_SN(data, k):
    """
    Calcul de la log-vraisemblance pour un point de rupture k donné.
    Modèle: X_t ~ SN(mu1, sigma1, theta) si t <= k
            X_t ~ SN(mu2, sigma2, theta) si t > k
    On estime les paramètres par MLE sur chaque sous-échantillon.
    """
    data = numpy.array(data, dtype=float)
    n = len(data)

    if k < 4 or k > n - 4:
        return -1e15, None, None

    data1 = data[:k]
    data2 = data[k:]

    # Estimation sur chaque sous-échantillon
    xi1, tau1, theta1 = estimation_MME(data1)
    xi2, tau2, theta2 = estimation_MME(data2)

    # Utiliser le theta moyen (contrainte theta commun)
    theta_commun = (len(data1) * theta1 + len(data2) * theta2) / n

    # Log-vraisemblance totale avec theta commun
    def calcul_ll(sub_data, xi, tau, theta):
        if tau <= 0:
            return -1e15
        z = (sub_data - xi) / tau
        ll = numpy.sum(numpy.log(2.0 / tau + 1e-300) +
                       norm.logpdf(z) +
                       numpy.log(norm.cdf(theta * z) + 1e-300))
        return ll

    ll1 = calcul_ll(data1, xi1, tau1, theta_commun)
    ll2 = calcul_ll(data2, xi2, tau2, theta_commun)

    params1 = (xi1, tau1, theta_commun)
    params2 = (xi2, tau2, theta_commun)

    return ll1 + ll2, params1, params2


def detection_rupture_SN(data):
    """
    Recherche du point de rupture k* qui maximise la log-vraisemblance.
    Retourne: k_hat, params1, params2, log_vraisemblances
    """
    data = numpy.array(data, dtype=float)
    n = len(data)

    log_vraisemblances = []
    k_range = list(range(4, n - 3))

    for k in k_range:
        ll, _, _ = log_vraisemblance_SN(data, k)
        log_vraisemblances.append(ll)

    if len(log_vraisemblances) == 0:
        return None, None, None, []

    idx_max = numpy.argmax(log_vraisemblances)
    k_hat = k_range[idx_max]

    # Recalculer les paramètres au point optimal avec MLE 
    data1 = data[:k_hat]
    data2 = data[k_hat:]
    xi1, tau1, theta1 = estimation_MLE(data1)
    xi2, tau2, theta2 = estimation_MLE(data2)
    theta_commun = (len(data1) * theta1 + len(data2) * theta2) / len(data)
    params1 = (xi1, tau1, theta_commun)
    params2 = (xi2, tau2, theta_commun)

    return k_hat, params1, params2, log_vraisemblances


# TEST DE KOLMOGOROV-SMIRNOV

def test_KS(data, k_hat):
    """
    Test de Kolmogorov-Smirnov pour tester l'existence de la rupture.
    H0: pas de rupture (les deux sous-échantillons suivent la même loi)
    H1: il y a une rupture
    """
    data = numpy.array(data, dtype=float)
    data1 = data[:k_hat]
    data2 = data[k_hat:]
    statistic, pvalue = ks_2samp(data1, data2)
    return statistic, pvalue


# GRAPHIQUES EN BASE64 POUR FLASK

def graphe_serie64(X, Date, choix, nom_variable="Nombre de trains annulés"):
    """Graphe de la série temporelle avec valeur moyenne"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)

    plt.plot(X, lw=3, color=numpy.random.rand(3))
    moyenne_empirique = sum(X) / len(X)
    plt.axhline(y=moyenne_empirique, color='r', linestyle='-', label="Valeur moyenne")

    indices = [i for i in range(len(X)) if i % max(1, math.ceil(len(X) / 20)) == 0]
    XX_ticks = [Date[i] for i in indices]
    plt.xticks(indices, XX_ticks, rotation=45, fontsize=9)

    plt.xlabel("Date")
    plt.ylabel(nom_variable)
    plt.title("Données de la région : %s" % choix)
    plt.legend(loc="best")
    plt.grid()

    plt.savefig(buf := BytesIO(), format='png', bbox_inches='tight')
    plt.close()
    return b64encode(buf.getvalue()).decode()


def graphe_rupture64(X, Date, k_hat, params1, params2, choix, nom_variable="Nombre de trains annulés"):
    """Graphe de la série avec la ligne de rupture"""
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    ax.set_facecolor((0.97, 0.97, 0.97))
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)

    plt.plot(X, lw=2, color='#2c3e50', label=nom_variable)

    # Ligne de rupture
    plt.axvline(x=k_hat, color='red', linestyle='--', lw=3,
                label=r"Rupture $\hat{k}=%d$" % k_hat)

    # Moyennes par segment
    m1 = numpy.mean(X[:k_hat])
    m2 = numpy.mean(X[k_hat:])
    plt.axhline(y=m1, xmin=0, xmax=k_hat / len(X), color='blue',
                linestyle='-', lw=2, label=r"$\hat{\mu}_1=%.1f$" % params1[0])
    plt.axhline(y=m2, xmin=k_hat / len(X), xmax=1, color='green',
                linestyle='-', lw=2, label=r"$\hat{\mu}_2=%.1f$" % params2[0])

    indices = [i for i in range(len(X)) if i % max(1, math.ceil(len(X) / 20)) == 0]
    XX_ticks = [Date[i] for i in indices]
    plt.xticks(indices, XX_ticks, rotation=45, fontsize=9)

    plt.xlabel("Date", fontsize=12)
    plt.ylabel(nom_variable, fontsize=12)
    plt.title("Détection de rupture — Région : %s" % choix, fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(buf := BytesIO(), format='png', bbox_inches='tight')
    plt.close()
    return b64encode(buf.getvalue()).decode()


def graphe_log_vraisemblance64(log_vraisemblances, k_hat, choix):
    """Graphe de la log-vraisemblance en fonction de k"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_facecolor((0.97, 0.97, 0.97))
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)

    k_range = list(range(4, 4 + len(log_vraisemblances)))
    plt.plot(k_range, log_vraisemblances, lw=2, color='#2c3e50')
    plt.axvline(x=k_hat, color='red', linestyle='--', lw=3,
                label=r"$\hat{k}=%d$" % k_hat)

    plt.xlabel(r"$k$", fontsize=15)
    plt.ylabel(r"$\Lambda_{n}(k)$", fontsize=15)
    plt.title("Log-vraisemblance — Région : %s" % choix, fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(alpha=0.3)

    plt.savefig(buf := BytesIO(), format='png', bbox_inches='tight')
    plt.close()
    return b64encode(buf.getvalue()).decode()


def graphe_densite_ajustee64(X, k_hat, params1, params2, choix):
    """Histogramme avec densités Skew Normal ajustées superposées"""
    data1 = numpy.array(X[:k_hat], dtype=float)
    data2 = numpy.array(X[k_hat:], dtype=float)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.gca()
    ax.set_facecolor((0.95, 0.95, 0.95))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Histogrammes normalisés
    all_data = numpy.array(X, dtype=float)
    bins = numpy.linspace(all_data.min(), all_data.max(), 20)

    ax.hist(data1, bins=bins, density=True, alpha=0.5, color='#3498db',
            edgecolor='white', label=r"Avant rupture ($t \leq \hat{k}$)")
    ax.hist(data2, bins=bins, density=True, alpha=0.5, color='#e74c3c',
            edgecolor='white', label=r"Après rupture ($t > \hat{k}$)")

    # Densités ajustées
    x_plot = numpy.linspace(all_data.min() - 5, all_data.max() + 5, 300)

    xi1, tau1, theta = params1
    xi2, tau2, _ = params2

    y1 = [densite_skew_normal(x, xi1, tau1, theta) for x in x_plot]
    y2 = [densite_skew_normal(x, xi2, tau2, theta) for x in x_plot]

    ax.plot(x_plot, y1, lw=3, color='#2980b9',
            label=r"$SN(\hat{\mu}_1=%.1f, \hat{\sigma}_1=%.1f, \hat{\theta}=%.2f)$" % (xi1, tau1, theta))
    ax.plot(x_plot, y2, lw=3, color='#c0392b',
            label=r"$SN(\hat{\mu}_2=%.1f, \hat{\sigma}_2=%.1f, \hat{\theta}=%.2f)$" % (xi2, tau2, theta))

    plt.xlabel("Valeurs", fontsize=13)
    plt.ylabel("Densité", fontsize=13)
    plt.title("Densités Skew Normal ajustées — Région : %s" % choix, fontsize=14)
    plt.legend(loc="best", fontsize=10)

    plt.savefig(buf := BytesIO(), format='png', bbox_inches='tight')
    plt.close()
    return b64encode(buf.getvalue()).decode()