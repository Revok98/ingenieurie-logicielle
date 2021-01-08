import json
import sklearn.metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Calcul et affichage des statistiques complètes de chaque classe en faisant apparître cette fois le F2 Score
def compute_scores(y_pred,y_true,lst_classe) :
    scores = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, beta = 2, labels = lst_classe,average=None)
    print("Classe               Precision   Recall    F2Score")
    print("--------------------------------------------------")
    for i in range(len(lst_classe)) :
        print('{0:17s}|   {1:2f}    {2:2f}  {3:2f}'.format(lst_classe[i],scores[0][i],scores[1][i],scores[2][i]))
    return

#Fonction d'affichage de la matrice de confusion
def plot_confusion_matrix(y_pred, y_true, lst_classe) :
    #On récupère la matrice de confusion calculée par sklearn
    cm = np.array(sklearn.metrics.confusion_matrix(y_true, y_pred))
    cm = cm / cm.astype(np.float).sum(axis=1)[:,None]
    ax = plt.subplot()
    sns.heatmap(cm, annot = True, ax=ax)
    # Affichage des labels et du titre
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(lst_classe)
    # Inversion de l'affichage pour le rendre plus lisible par un humain
    lst_classe.reverse()
    ax.yaxis.set_ticklabels(lst_classe)
    return

# Calcul et affichage de la courbe ROC-AUC et détermination du threshold
def compute_roc_auc_curve(lst_classe, pas, y_pred, y_true) :
    tot = len(y_pred)
    dic = {}
    #Instanciation du dictionnaire qui permettra de calculer la courbe
    for i in lst_classe :
        dic[i] = [[],[]]
    n_classes = lst_classe
    ind = 0
    # Pas du threshold
    nb = int(0.99/pas)
    rng = np.flip(np.linspace(0.000,1,nb))
    gmeans = [0 for i in range(len(rng))]
    for th in range(len(rng)) :
        threshold = rng[th]
        maxi = 'irrelevant'
        #Instanciation du nombre de TP et de FP
        for i in lst_classe :
            dic[i][0].append(0)
            dic[i][1].append(0)
        for l in range(len(y_pred)) :
            lst_predicted = []
            for j in lst_classe :
                if(y_pred[l][j] >= threshold) :
                    lst_predicted.append(j)
            for j in lst_predicted :
                if(y_true[l] == j) :
                    dic[j][0][ind] += 1
                else :
                    dic[j][1][ind] += 1
        ind += 1
    # Récupération des données calculées et on remet dans l'intervalle voulu les valeurs
    for i in lst_classe :
        maxi = max(dic[i][0])
        maxi2 = max(dic[i][1])
        dic[i][0] = [1 if maxi == 0 else dic[i][0][j]/maxi for j in range(len(dic[i][0]))]
        dic[i][1] = [1 if maxi2 == 0 else dic[i][1][j]/maxi2 for j in range(len(dic[i][1]))]
        for ind in range(len(dic[i][0])) :
            gmeans[ind] += dic[i][0][ind] * (1-dic[i][1][ind])
        to_plot_x = dic[i][1]
        to_plot_y = dic[i][0]
        plt.legend(lst_classe)
        plt.title("ROC_AUC Curve")
        plt.plot(to_plot_x,to_plot_y)
    # Affichage de la courbe de threshold.
    gmeans = [x/len(lst_classe) for x in gmeans]
    to_plot_x = rng
    to_plot_y = gmeans
    index = np.argmax(gmeans)
    print("Best threshold :" + str(rng[index]))
    plt.figure()
    plt.plot(to_plot_x,to_plot_y)
    plt.title("Courbe d'évolution de la précision et du recall en fonction du threshold")


# Affichage et calcul de la courbe ROC-AUC dans laquelle on considère que prédire irrelevant est équivalent à une bonne rédiction
def Cleymevin_curve(lst_classe, pas, y_pred, y_true) :
    tot = len(y_pred)
    dic = {}
    #Instanciation du dictionnaire qui permettra de calculer la courbe
    for i in lst_classe :
        dic[i] = [[],[]]
    n_classes = lst_classe
    ind = 0
    # Pas du threshold
    nb = int(0.99/pas)
    rng = np.flip(np.linspace(0.000,1,nb))
    gmeans = [0 for i in range(len(rng))]
    for th in range(len(rng)) :
        threshold = rng[th]
        maxi = 'irrelevant'
        #Instanciation du nombre de TP et de FP
        for i in lst_classe :
            dic[i][0].append(0)
            dic[i][1].append(0)
        # Pour chacune des prédictions on remplit soit TP soit FP
        for l in range(len(y_pred)) :
            lst_predicted = []
            for j in lst_classe :
                if(y_pred[l][j] >= threshold) :
                    lst_predicted.append(j)
            for j in lst_predicted :
                if(y_true[l] == j or j =='irrelevant') :
                    dic[y_true[l]][0][ind] += 1
                else :
                    dic[j][1][ind] += 1
        ind += 1
    # Récupération des données calculées et on remet dans l'intervalle voulu les valeurs
    for i in lst_classe :
        maxi = max(dic[i][0])
        maxi2 = max(dic[i][1])
        dic[i][0] = [1 if maxi == 0 else dic[i][0][j]/maxi for j in range(len(dic[i][0]))]
        dic[i][1] = [1 if maxi2 == 0 else dic[i][1][j]/maxi2 for j in range(len(dic[i][1]))]
        for ind in range(len(dic[i][0])) :
            gmeans[ind] += dic[i][0][ind] * (1-dic[i][1][ind])
        to_plot_x = dic[i][1]
        to_plot_y = dic[i][0]
        plt.legend(lst_classe)
        plt.title("Cleymevin Curve")
        plt.plot(to_plot_x,to_plot_y)
    # Affichage de la courbe de threshold.
    gmeans = [x/len(lst_classe) for x in gmeans]
    index = np.argmax(gmeans)
    print("Best threshold :" + str(rng[index]))
    to_plot_x = rng
    to_plot_y = gmeans
    plt.figure()
    plt.title("Evolution de la précision/recall en fonction du threshold avec irrelevant TP")
    plt.plot(to_plot_x,to_plot_y)