import sklearn.metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def compute_scores(y_pred,y_true,lst_classe) :
    scores = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, beta = 2, labels = lst_classe,average=None)
    print("Classe               Precision   Recall    F2Score")
    print("--------------------------------------------------")
    for i in range(len(lst_classe)) :
        print('{0:17s}|   {1:2f}    {2:2f}  {3:2f}'.format(lst_classe[i],scores[0][i],scores[1][i],scores[2][i]))
    return

def plot_confusion_matrix(y_pred, y_true, lst_classe) :
    cm = np.array(sklearn.metrics.confusion_matrix(y_true, y_pred))
    cm = cm / cm.astype(np.float).sum(axis=1)[:,None]
    """
    for i in range(len(cm)):
        cm[i] = np.sum[cm]
    """
    ax = matplotlib.pyplot.subplot()
    sns.heatmap(cm, annot = True, ax=ax)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(lst_classe)
    lst_classe.reverse()
    ax.yaxis.set_ticklabels(lst_classe)
    return

def compute_roc_auc_curve(lst_classe, pas, y_pred, y_true) :
    tot = len(y_pred)
    dic = {}
    for i in lst_classe :
        dic[i] = [[],[]]
    n_classes = lst_classe
    ind = 0
    nb = int(0.99/pas)
    rng = np.flip(np.linspace(0.000,1,nb))
    gmeans = [0 for i in range(len(rng))]
    for th in range(len(rng)) :
        threshold = rng[th]
        maxi = 'irrelevant'
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
        plt.plot(to_plot_x,to_plot_y)
    gmeans = [x/len(lst_classe) for x in gmeans]
    to_plot_x = rng
    to_plot_y = gmeans
    index = np.argmax(gmeans)
    print("Best threshold :" + str(rng[index]))
    plt.figure()
    plt.plot(to_plot_x,to_plot_y)
    #plt.show()
        #Plus qu'à faire l'affichage des courbes. La fonction Cleymevin devrait pas mal ressembler d'ailleurs

def Cleymevin_curve(lst_classe, pas, y_pred, y_true) :
    tot = len(y_pred)
    dic = {}
    for i in lst_classe :
        dic[i] = [[],[]]
    n_classes = lst_classe
    ind = 0
    nb = int(0.99/pas)
    rng = np.flip(np.linspace(0.000,1,nb))
    gmeans = [0 for i in range(len(rng))]
    for th in range(len(rng)) :
        threshold = rng[th]
        maxi = 'irrelevant'
        for i in lst_classe :
            dic[i][0].append(0)
            dic[i][1].append(0)
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
        plt.plot(to_plot_x,to_plot_y)
    gmeans = [x/len(lst_classe) for x in gmeans]
    index = np.argmax(gmeans)
    print("Best threshold :" + str(rng[index]))
    to_plot_x = rng
    to_plot_y = gmeans
    plt.figure()
    plt.plot(to_plot_x,to_plot_y)