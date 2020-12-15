import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import os
import numpy as np
import pyautogui
import time


def gather_data(complete_file, update=False):
    """ Retravaille les données collectées pour les rendre exploitables."""
    ignore_files()
    files_list = [file for file in os.listdir() if file[-1] == "'"]
    y1, y2, final_scores_list = final_scores(files_list)
    scoreless = find_scoreless(final_scores_list, files_list)
    data_all = pd.DataFrame()
    for file in files_list:
        if (file not in final_scores_list.keys()
            and file not in scoreless):
            data_all = pd.concat([data_all, pd.read_csv(file)])
            data_all.reset_index(drop=True, inplace=True)
    gathered_data = complete_data(data_all,
                                  final_scores_list, update, complete_file, y1)
    gathered_data.to_csv(complete_file)
    to_git(['med.py', 'modelization.py', '.gitignore', 'alldata'])


def location(texte, car):
    """Donne la position d'un caractère dans un string"""
    list_ind = []
    for i, j in enumerate(texte):
        if j == car:
            list_ind.append(i)
    return list_ind


def final_scores(files_list):
    """Renvoie la liste des matchs pour lesquels on a l'info sur le score
    final, et isole dans deux dataframes les lignes donnant le score final"""
    games_list = {}
    for i in range(len(files_list)):
        pos_minute = location(files_list[i], '_')[1]
        match = files_list[i][:pos_minute]
        if match not in games_list.keys():
            games_list[match] = files_list[i]
        else:
            cur_minute = int(files_list[i][pos_minute + 1: -1])
            last_minute = int(games_list[match][pos_minute + 1: -1])
            if cur_minute > last_minute:
                games_list[match] = files_list[i]

    y1 = pd.DataFrame(columns=['Match', 'Nb_buts'])
    y2 = pd.DataFrame(columns=['Match', 'Nb_buts'])
    for i, match in enumerate(games_list):
        y1.loc[i, 'Match'] = pd.read_csv(games_list[match]).iloc[0]['Match']
        y1.loc[i, 'Nb_buts'] = pd.read_csv(games_list[match]).iloc[0]['Buts']
        y2.loc[i, 'Match'] = pd.read_csv(games_list[match]).iloc[1]['Match']
        y2.loc[i, 'Nb_buts'] = pd.read_csv(games_list[match]).iloc[1]['Buts']

    return y1, y2, games_list


def find_scoreless(dictionnary, files_list):
    """Liste les matchs pour lesquels le score final est incertain"""
    scoreless = []
    for elem in dictionnary.keys():
        match = pd.read_csv(dictionnary[elem])
        if int(match.loc[0, 'Minute'][0:2]) < 89:
            for file in files_list:
                if file == dictionnary[elem]:
                    scoreless.append(file)
    return scoreless


def complete_data(dataset, games_list, update, complete_file, y1):
    """Complète le jeu de données en calculant pour chaque match et chaque
    minute, les statistiques les plus probables en utilisant celles obtenues
    à des minutes proches"""
    dataset_new = dataset
    if update:
        already_compl = list(y1['Match'].unique())
        games_list_new = {}
        for game in games_list.keys():
            url = pd.read_csv(games_list[game]).loc[0, 'Match']
            if url not in already_compl:
                games_list_new[game] = games_list[game]
        games_list = games_list_new
    for game in games_list.keys():
        print(game)
        min_with_stats = {}
        for row in range(dataset.shape[0]):
            if dataset.iloc[row, 1] == pd.read_csv(games_list[game]).iloc[0, 1]:
                row_minute = int(dataset.iloc[row, 0][:-1])
                min_with_stats[row_minute] = row
        list_min_with_stats = list(min_with_stats.keys())
        list_min_with_stats.sort()
        # print(list_min_with_stats)
        for i in range(20, 90):
            if i not in list_min_with_stats:
                count = 0
                while (i > list_min_with_stats[count] - 1
                       and count < len(list_min_with_stats) - 1):
                    count += 1
                if count < len(list_min_with_stats):
                    min_bornes = [list_min_with_stats[count - 1],
                                  list_min_with_stats[count]]
                    index_bornes = [min_with_stats[min_bornes[0]],
                                    min_with_stats[min_bornes[1]]]
                    game_link = dataset.loc[index_bornes[0], 'Match']
                    values_nl_1 = [str(i) + "'",
                                   game_link,
                                   dataset.loc[index_bornes[0], 'Equipe']]
                    values_nl_2 = [str(i) + "'",
                                   game_link,
                                   dataset.loc[index_bornes[0] + 1, 'Equipe']]
                    for j in range(18):
                        v_inf_1 = dataset.loc[index_bornes[0],
                                              dataset.columns.to_list()[j + 3]]
                        v_sup_1 = dataset.loc[index_bornes[1],
                                              dataset.columns.to_list()[j + 3]]
                        v_inf_2 = dataset.loc[index_bornes[0] + 1,
                                              dataset.columns.to_list()[j + 3]]
                        v_sup_2 = dataset.loc[index_bornes[1] + 1,
                                              dataset.columns.to_list()[j + 3]]
                        # print(dataset.loc[index_bornes[0], 'Match'])
                        gap_time = min_bornes[1] - min_bornes[0]
                        prop_time = (i - min_bornes[0]) / gap_time
                        values_nl_1.append(np.floor(prop_time
                                                    * (v_sup_1 - v_inf_1)
                                                    + v_inf_1))
                        values_nl_2.append(np.floor(prop_time 
                                                    * (v_sup_2 - v_inf_2)
                                                    + v_inf_2))
                    dataset_new.loc[dataset_new.shape[0]] = values_nl_1
                    dataset_new.loc[dataset_new.shape[0]] = values_nl_2
    dataset_new.dropna()
    for row in range(dataset.shape[0]):
        dataset_new['Time'] = dataset_new.loc[row, 'Minute'][0:len(dataset_new.loc[row, 'Minute']) - 1]

    return dataset_new


def ignore_files():
    fichier = open(".gitignore", "w", encoding='utf-8')
    fichier.close()
    for file in os.listdir():
        if file not in ['alldata', '.git',
                        '.gitignore', 'med.py', 'modelization.py']:
            fichier = open(".gitignore", "a", encoding='utf-8')
            fichier.write('\n' + file)
            fichier.close()


def to_git(to_update):
    """Met à jour automatiquement les fichiers voulus sur github"""

    passfile = open('password_github.txt', 'r', encoding='utf-8')
    passphrase = passfile.read()
    passfile.close()

    directory = "'Documents/ENSAE/Python pour Data Scientist/TimeToBet'"
    ssh_key = "'git@github.com:BlancoLouis/TimeToBet.git'"

    pyautogui.press('win')
    time.sleep(2)
    pyautogui.write('git bash')
    time.sleep(2)
    pyautogui.press('enter')
    time.sleep(5)
    pyautogui.write("cd " + directory)
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.write("git remote set-url " + ssh_key)
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.write("git status")
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.write("git pull origin master")
    pyautogui.press('enter')
    time.sleep(5)
    pyautogui.write(passphrase)
    pyautogui.press('enter')
    time.sleep(2)
    for file in to_update:
        pyautogui.write("git add " + file)
        pyautogui.press('enter')
        time.sleep(2)
    pyautogui.write("git commit -m 'Initial commit'")
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.write("git commit -m 'Initial commit'")
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.write("git push origin master")
    pyautogui.press('enter')
    time.sleep(5)
    pyautogui.write(passphrase)
    pyautogui.press('enter')
    time.sleep(4)
    pyautogui.write("exit")
    pyautogui.press('enter')


def sep_by_time(datasets, minute=None):
    """Permet de diviser un dataset en 70 datasets ne contenant
    chacun que les statistiques obtenues à une minute. Peut réaliser
    cela sur une liste dde datasets."""
    datasets_by_time = []
    for i in range(len(datasets)):
        datasets_by_time.append([])
    for j, elem in enumerate(datasets):
        if minute is None:
            for i in range(20, 90):
                data_sep = elem[elem['Minute'] == str(i) + "'"]
                datasets_by_time[j].append(data_sep)
        else:
            datasets_by_time[j].append(elem)
    return datasets_by_time


def sep_by_team(dataset):
    """Permet de diviser un dataset en 2 datasets contenant respectivement
    uniquement les lignes paires et impaires. On l'utilise ici pour séparer
    équipes jouant à domicile, et équipes jouant à l'extérieur,
    mais aussi pour retirer toute dépendance entre les lignes."""
    list_locs = [2 * x for x in range(int(dataset.shape[0] / 2))]
    df_1_lines = []
    df_2_lines = []
    for i in list_locs:
        df_1_lines.append(dataset.iloc[i])
        df_2_lines.append(dataset.iloc[i + 1])
    df_1 = pd.DataFrame(df_1_lines).fillna(0)
    df_2 = pd.DataFrame(df_2_lines).fillna(0)
    datasets_by_team = [df_1, df_2]
    return datasets_by_team


def regs(datafile, minute, team):
    """Crée un modèle de régression logistique multiclasse,
    entraîné sur un train set, testé sur un test set."""
    data = sep_by_time(sep_by_team(pd.read_csv(datafile)))

    X = data[team][minute - 20]
    y = pd.DataFrame()
    files_list = [file for file in os.listdir() if file[-1] == "'"]
    y1, y2, final_scores_list = final_scores(files_list)
    Y = [y1, y2]
    X = X.merge(Y[team], how='left', on='Match')
    y = X['Nb_buts'].astype('int')
    X = X.drop(['Unnamed: 0', 'Minute',
                'Match', 'Equipe', 'Nb_buts',
                'Remplacements'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print(X_train, X_test, y_train, y_test)

    log_reg = LogisticRegression(penalty='none',
                                 multi_class='multinomial',
                                 solver='newton-cg')

    stds = preprocessing.StandardScaler()
    Z_train = stds.fit_transform(X_train)
    Z_test = stds.fit_transform(X_test)
    log_reg.fit(Z_train, y_train)
    print(metrics.accuracy_score(y_train, log_reg.predict(Z_train)))
    print(metrics.accuracy_score(y_test, log_reg.predict(Z_test)))

    return log_reg, stds


def prepare_data(dataset, stds, minute):
    """Filtre et standardise les données qui seront envoyées
    au modèle de régression logistique multiclasses."""
    if dataset is None:
        return None

    dataset = dataset.drop(['Minute', 'Match',
                            'Equipe', 'Remplacements'], axis=1)
    data = sep_by_time(sep_by_team(dataset), minute)
    for i in range(2):
        for stat_time in data[i]:
            stds_ = stds[i]
            stat_time = stds_.fit_transform(stat_time)
    return data


def predict(dataset, file):
    """Prédit le score d'un match et sa probabilité
    à partir des statistiques de ce match et d'une base de données
    contenant des statistiques de matchs similaires."""
    minute = int(dataset.loc[0, 'Minute'][0:2])
    dataset['Time'] = minute
    t1_buts = dataset.loc[0, 'Buts']
    t2_buts = dataset.loc[1, 'Buts']
    models = [regs(file, minute, 0), regs(file, minute, 1)]
    data = prepare_data(dataset, [models[0][1], models[1][1]], minute)
    y1_scores = models[0][0].decision_function(data[0][0])[0]
    y2_scores = models[1][0].decision_function(data[1][0])[0]
    print(y1_scores, y2_scores)
    ns = likely_scores([y1_scores, y2_scores], [t1_buts, t2_buts])
    ns[0] = list(ns[0])
    ns[1] = list(ns[1])
    print(ns)
    prediction = [ns[0].index(max(ns[0])), ns[1].index(max(ns[1]))]
    probas = [max(ns[0]) / sum([x for x in ns[0] if x > 0]),
              max(ns[1]) / sum([x for x in ns[1] if x > 0])]
    return prediction, probas


def likely_scores(scores, buts):
    """Filtre certaines prédictions impossibles,
    notamment celles qui annoncent un nombre de buts plus
    faible que celui déjà atteint."""
    # new_score = lambda score, gap: score * ((1 + gap) ** (-1 / 2))
    for i, elem in enumerate(scores):
        for j, score in enumerate(elem):
            if j - buts[i] < 0:
                scores[i][j] = -100000
    return scores


# gather_data('alldata', True)
# sep_by_time([pd.read_csv('touteladonnée')])
# sep_by_team(pd.read_csv('touteladonnée'))
# regs('touteladonnée', 80, 0)
# print(predict(pd.read_csv("Atalanta_Fiorentina_69'"), 'alldata'))
# ignore_files()
to_git(['med.py', 'modelization.py', '.gitignore', 'alldata'])
