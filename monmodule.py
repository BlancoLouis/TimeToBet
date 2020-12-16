import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import os
import numpy as np
import pyautogui
import time
from selenium import webdriver
import dash
from urllib import request
import bs4
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import json
import re
import datetime
from pandas.util.testing import assert_frame_equal
from fuzzywuzzy import process


path_to_web_driver = "chromedriver"

BETCLIC_URL = 'https://www.betclic.com/fr/paris-sportifs/football-s1'

BET_URLS_DICT = {
    'L1':
    'https://www.betclic.com/fr/paris-sportifs/football-s1/ligue-1-uber-eats-c4',
    'L2':
    'https://www.betclic.com/fr/paris-sportifs/football-s1/ligue-2-bkt-c19',
    'Liga':
    'https://www.betclic.com/fr/paris-sportifs/football-s1/espagne-liga-primera-c7',
    'SerieA':
    'https://www.betclic.com/fr/paris-sportifs/football-s1/italie-serie-a-c6',
    'Bundesliga':
    'https://www.betclic.com/fr/paris-sportifs/football-s1/allemagne-bundesliga-c5',
}

stats_cats = ['Possession', 'Attaques', 'Attaques dangereuses',
              'Coups francs', 'Coups de pied arrêtés',
              'Buts', 'Tirs cadrés', 'Tirs non cadrés',
              'Tirs arrêtés', 'Tirs sur le poteau',
              'Pénaltys', 'Touches', 'Corners',
              'Hors-jeu', 'Fautes', 'Carton jaune',
              'Carton rouge', 'Remplacements']

MATCHENDIRECT_URLS_DICT = {
    'L1': 'https://www.matchendirect.fr/france/ligue-1/',
    'L2': 'https://www.matchendirect.fr/france/ligue-2/',
    'Liga': 'https://www.matchendirect.fr/espagne/primera-division/',
    'SerieA': 'https://www.matchendirect.fr/italie/serie-a/',
    'Bundesliga': 'https://www.matchendirect.fr/allemagne/bundesliga-1/',
}


def select_champ(other_champs=None, champs=None):
    """Boucle sur tous les championnats choisis pour établir
    la liste de ceux qui ont des matchs actuellement en direct"""
    champs_links = {}
# on regarde si l'utilisateur veut ajouter un championnat à la liste de base
    if other_champs is not None:
        for elem in other_champs.keys():
            champs_links[elem] = other_champs[elem]
# on regarde si l'utilisateur veut se focaliser uniquement
# sur certains championnats de la liste de base
    if champs is not None:
        if champs != 0:
            for elem in champs:
                champs_links[elem] = MATCHENDIRECT_URLS_DICT[elem]
    else:
        for elem in MATCHENDIRECT_URLS_DICT.keys():
            champs_links[elem] = MATCHENDIRECT_URLS_DICT[elem]
    count_no_match = 0
    for champ in champs_links.keys():
        chrome_options = webdriver.ChromeOptions()
        browser = webdriver.Chrome(executable_path=path_to_web_driver,
                                   options=chrome_options)
        browser.get(champs_links[champ])
        time.sleep(2)
        try:
            browser.find_element_by_id('onetrust-accept-btn-handler').click()
        except:
            pass
        games_list = browser.find_elements_by_class_name('sl')
        if len(games_list) > 0:
            count_no_match += 1
            select_game(len(games_list), browser)
        browser.quit()

    if count_no_match == 0:
        time.sleep(30)
        browser.quit()
        return
    return


def select_game(size, browser_origin):
    """Boucle sur un championnat pour ouvrir
    les pages des matchs en direct de ce championnat."""
    browser = browser_origin
    for i in range(size):
        games_list = browser.find_elements_by_class_name('sl')
        if i < size:
            link = games_list[i].find_element_by_css_selector('a').get_attribute('href')
            infos_game(link)
    browser.quit()
    return


def infos_game(link=None, to_csv=True):
    """Depuis la page matchendirect d'un match précis, on récupère toutes
    les statistiques disponibles et on les enregistre dans un fichier csv."""

    if link is not None:
        chrome_options = webdriver.ChromeOptions()
        browser = webdriver.Chrome(executable_path=path_to_web_driver,
                                   options=chrome_options)
        browser.get(link)

    stats_table = browser.find_element_by_xpath('//*[@id="ajax-match-detail-3"]/div[1]/div[2]/table/tbody')
    stats = stats_table.find_elements_by_css_selector('td')

    minute = browser.find_element_by_xpath('//*[@id="ajax-match-detail-1"]/div/div[3]/div[2]/div').text
    team_1 = browser.find_element_by_xpath('//*[@id="ajax-match-detail-1"]/div/div[3]/div[1]/a').text
    team_2 = browser.find_element_by_xpath('//*[@id="ajax-match-detail-1"]/div/div[3]/div[3]/a').text 
    index = pd.MultiIndex.from_product([[minute], [link], [team_1, team_2]],
                                       names=['Minute', 'Match', 'Equipe'])
    t_1 = {}
    t_2 = {}
    for elem in stats_cats:
        t_1[elem] = 0
        t_2[elem] = 0
    # print(int(len(stats) / 5))
    if int(len(stats) / 5) > 7:
        for i in range(int(len(stats) / 5)):
            t_1[stats[5 * i + 2].text] = stats[5 * i].text
            t_2[stats[5 * i + 2].text] = stats[5 * i + 4].text
        game_stats = pd.DataFrame(
            data=[list(t_1.values()),
                  list(t_2.values())],
            index=index,
            columns=stats_cats)
        browser.quit()
    else:
        game_stats = None
    if game_stats is not None and to_csv:
        csv_file_name = str(team_1) + "_" + str(team_2) + '_' + str(minute)
        game_stats.to_csv(csv_file_name)
    return(game_stats)


def get_stats(other_champs=None, champs=None):
    """ Récupère une fois par minute les statistiques
    des matchs en direct des championnats sélectionnés."""
    start_time = time.time()
    select_champ(other_champs, champs)
    end_time = time.time()
    duration = end_time - start_time
    if duration < 60:
        time.sleep(60 - duration)
    get_stats(other_champs, champs)


def red_card():
    """Affiche tous les matchs déjà étudiés dans lesquels
    un carton rouge a été distribué."""
    files_list = [file for file in os.listdir() if file[-1] == "'"]
    links = set()
    for file in files_list:
        links.add(pd.read_csv(file).loc[0, 'Match'])
    print(links)
    for link in links:
        chrome_options = webdriver.ChromeOptions()
        browser = webdriver.Chrome(executable_path=path_to_web_driver,
                                   options=chrome_options)
        browser.get(link)
        stats_table = browser.find_element_by_xpath('//*[@id="ajax-match-detail-3"]/div[1]/div[2]/table/tbody')
        stats = stats_table.find_elements_by_css_selector('td')
        for elem in stats:
            if elem.text == 'Carton rouge':
                print(link)
        browser.quit()


def gather_data(complete_file, update=False, to_git_or_not_to_git=True):
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
    if to_git_or_not_to_git :
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
    dataset = dataset_new.dropna()
    for row in range(dataset_new.shape[0]):
        tim = dataset_new.loc[row, 'Minute'][0:len(dataset_new.loc[row, 'Minute']) - 1]
        dataset_new.loc[row, 'Temps'] = tim
        print(dataset_new.loc[row, 'Temps'])

    return dataset_new


def ignore_files():
    fichier = open(".gitignore", "w", encoding='utf-8')
    fichier.close()
    for file in os.listdir():
        if file not in ['alldata',
                        '.git',
                        '.gitignore',
                        'forignore.txt',
                        'monmodule.py']:
            fichier = open(".gitignore", "a", encoding='utf-8')
            fichier.write('\n' + file)
            fichier.close()
    content = open("forignore.txt", "r", encoding='utf-8')
    for line in content:
        fichier = open(".gitignore", "a", encoding='utf-8')
        fichier.write('\n' + line)
        fichier.close()



def to_git(to_update, directory, ssh_key, passphrase_file):
    """Met à jour automatiquement les fichiers voulus sur github"""

    passfile = open(passphrase_file, 'r', encoding='utf-8')
    passphrase = passfile.read()
    passfile.close()

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
    entraîné sur un train set standardisé, testé sur un test
    set standardisé."""
    data = sep_by_time(sep_by_team(pd.read_csv(datafile)))

    Y = final_scores_2(datafile)
    X = data[team][minute - 20]
    X = X.drop(['Unnamed: 0',
                'Minute',
                'Match',
                'Equipe',
                'Remplacements'],
               axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y[team])
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


def final_scores_2(datafile):
    data = pd.read_csv(datafile)
    games_list_index = {}
    games_list = list(data['Match'].unique())
    for elem in games_list:
        games_list_index[elem] = 0
    for elem in games_list:
        for row in range(data.shape[0]):
            if (data.loc[row, 'Match'] == elem
                and data.loc[row, 'Temps'] > data.loc[games_list_index[elem], 'Temps']):
                games_list_index[elem] = row
    y1_lines = []
    y2_lines = []
    for row in range(data.shape[0]):
        if row in games_list_index.values():
            y1_lines.append(data.iloc[row])
            y2_lines.append(data.iloc[row + 1])
    y1 = pd.DataFrame(y1_lines)['Buts'].astype(int)
    y2 = pd.DataFrame(y2_lines)['Buts'].astype(int)
    return [y1, y2]


def predict(dataset, file):
    """Prédit le score d'un match et sa probabilité
    à partir des statistiques de ce match et d'une base de données
    contenant des statistiques de matchs similaires."""
    minute = dataset.loc[0, 'Minute'][0:len(dataset.loc[0, 'Minute']) - 1]
    if len(minute) > 3:
        minute = 45
    else :
        minute = int(minute)
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
    new_score = lambda score, gap: score * ((1 + gap) ** (-1 / 2))
    for i, elem in enumerate(scores):
        for j, score in enumerate(elem):
            if j - buts[i] < 0:
                scores[i][j] = -100000
            else:
                scores[i][j] = new_score(score, j - buts[i])
    return scores


def process_url(url):
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    request_text = request.urlopen(req).read()
    bet_page = bs4.BeautifulSoup(request_text, 'lxml')
    return bet_page


def fetch_bet_urls(championship_url):
    """Pour l'url Betclic d'un championnat de football donné,
    fetch_bet_urls renvoie tous les URLs des matchs en direct"""
    request_text = request.urlopen(championship_url).read()
    bet_page = bs4.BeautifulSoup(request_text, 'lxml')
    live_bet_game = []
    for elem in bet_page.findAll('app-live-event'):
        live_bet_game.append(elem.find('a').get('href'))
    live_bet_url = ['https://www.betclic.com' + elem for elem in live_bet_game]

# Cette étape semble peu inutile à première vue mais permet d'éviter un problème
# dans la situation où on boucle sur la liste d'URLs qu'on obtient via cette
# fonction et où on obtient un seul URL (un seul match en direct).
# Sans le 'if' qui arrive, notre code penserait que la liste d'URLs
# est composée des éléments 'h', 't', 't', 'p'...

    if len(live_bet_url) == 1:
        live_bet_url = [
            live_bet_url[0],
        ]
    return live_bet_url


def get_game_odd(game_url):
    """Cette fonction nous donne, pour un URL de match donné, les côtes de
    Score exact de ce match. Sur Betclic, la fonction qui gère ces côtes est
    régie par un code JS pour pouvoir être actualisée en continu."""
    print(f'Fetching odds from game_url={game_url}')
    game_page = process_url(game_url)
    script = game_page.find('script', text=re.compile('Score exact'))
    if script is not None:
        print(f'Found script {len(script.get_text())} characters long in {game_url}')
        # previous behaviour clean_json = script.get_text()
        clean_json = str(script)
        begin = clean_json.find('{')
        clean_json = clean_json[begin:-9]
        odds = {}
        payload_content = json.loads(clean_json)
        for item in payload_content['body']['markets']:
            if item['name'] == 'Score exact':
                for item2 in item['selections']:
                    odds[item2['name']] = float(item2['odds'])
        return odds

# Parfois (quand il vient d'y avoir un but par exemple), les côtes
# ne sont pas disponibles car le site les recalcule. Dans ce cas,
# on récupère un dictionnaire vide.

    else:
        return {}


def get_game_teams(game_url):
    """Renvoie une liste composée des noms des deux équipes
    (avec celle qui joue à domicile en 1er)."""
    print(f'Fetching teams for game_url={game_url}')
    game_page = process_url(game_url)
    game = game_page.find('title').get_text()
    sep = game.find(' - ')
    end = game.find('|')
    team1, team2 = game[: sep], game[sep + 3: end - 1]
    print(f'Found teams={[team1, team2]} for game_url={game_url}')
    return team1, team2

# On pourrait sûrement construire plus directement cette fonction à partir
# de la précédente.


def get_game_name(game_url):
    both_teams = get_game_teams(game_url)
    team1 = both_teams[0]
    team2 = both_teams[1]
    return team1 + ' - ' + team2


def get_game_time(game_url):
    print(f'Fetching game time for game_url={game_url}')
    game_page = process_url(game_url)
    score_board = game_page.find('span', {'class': 'liveScoreboard_dateTime'})
    if score_board is None:
        return('NA')
    game_time = score_board.get_text().replace('\n', '')
    if game_time == '':
        print(f'Could not find a game time for game_url={game_url}')
        return 'NA'
    game_time = game_time[: game_time.find("'")]
    print(f'Found game_time={game_time}')
    return game_time.lstrip()


def get_odds(game_url):
    """Cette fonction crée un DataFrame avec les côtes (score exact toujours)
    du match."""

    try:
        game_time = get_game_time(game_url)
        game_name = get_game_name(game_url)
    except RuntimeError as error:
        print(f'Could not fetch informations from game_url={game_url} reason: {error}')
        raise
    index = pd.MultiIndex.from_product(
        [[game_time], [game_name]],
        names=['Minute', 'Match'],
    )
    game_odds = pd.DataFrame(data=get_game_odd(game_url), index=index)
    return game_odds


def create_odds_file(
    list_of_champ_links: list,
    threshold=4,
    count_file=0,
    file_dict={},
    file_nb=0
):
    """Cette fonction récupère les côtes d'un match chaque minute pendant
    n minutes, avec n = threshold, puis, une fois qu'il a fini, crée un
    fichier csv par match avec une ligne correspondant aux côtes pour une
    minute du match.
    file_nb correspond au nombre de fois où l'on veut répéter cette opération
    (qui dure threshold minutes), ainsi cette fonction doit durer
    threshold * file_nb minutes et crée un nombre file_nb de fichiers.
    Notons que le fichier file_nb = k + 1 contient toutes les lignes du fichier
    file_nb = k)."""

    start_time = time.time()
    print(f'start={start_time} looking up: {list_of_champ_links}')
    for champ_link in list_of_champ_links:
        bet_urls = fetch_bet_urls(champ_link)
        print(f'Found {len(bet_urls)} bet(s) for champ_link={champ_link}')
        for game in bet_urls:
            print(f'Processing game={game} for champ_link={champ_link}')
            try:
                process_url(game)
            except Exception as err:
                print(f'Could not process game={game} reason: {err}')
                return
            try:
                file_dict[game]
            except:
                file_dict[game] = get_odds(game)
                # si file_dict[game] n'existe pas, on le crée
            else:
                file_dict[game] = pd.concat([file_dict[game], get_odds(game)])
                # sinon on lui ajoute les côtes qu'on vient de trouver
    print(f'Done processing {len(list_of_champ_links)} elements.')
    count_file += 1
    if count_file <= threshold:
        end_time = time.time()
        duration = end_time - start_time

# Si on a fini de récupérer toutes les infos qui nous intéressent en moins
# d'une minute, on attend avant de continuer

        if duration < 60:
            sleepy_time = 60 - duration
            print(f'Pausing a bit... (sleepy_time={sleepy_time})')
            time.sleep(sleepy_time)

# On utilise ici une fonction récursive.

    if count_file != threshold + 1:
        create_odds_file(
            list_of_champ_links,
            threshold=threshold,
            count_file=count_file,
            file_dict=file_dict,
            file_nb=file_nb,
        )
    else:
        for champ_link in list_of_champ_links:
            for game in fetch_bet_urls(champ_link):
                teams = get_game_teams(game)
                csv_filename = f"dataset_{'-vs-'.join(teams[0:])}_{file_nb}.csv"
                file_dict[game].to_csv(csv_filename)
    return


def main():
    for compter in range(10):
        temps = time.time()
        create_odds_file(
            [
              'https://www.betclic.com/fr/paris-sportifs/football-s1/argentine-torneo-a-c21273',
            ],
            file_nb=compter,
            threshold=10
        )
        if temps <= 300:
            sleepy_time = 300 - temps
            print(f'Going to sleep for {sleepy_time}s')
            time.sleep(sleepy_time)
            print('Resuming processing')

# DATA VISUALISATION

# On utilisera le module Dash, particulièrement utile pour créer des
# interactions entre les différents éléments qu'on organise sur le site
# créé. Les fonctions définies dans un premier temps servent de fonction
# intermédiaire pour récupérer nos données.


def get_live_teams(championship_url):
    """Pour un URL Betclic de championnat donné, cette fonction
    retourne une liste d'équipes sous un format utile pour la partie
    Data Visualisation."""
    list_live_teams = [live_team for game in fetch_bet_urls(championship_url)
                       for live_team in get_game_teams(game)]
    return list_live_teams


def get_ranking(championship_url):
    """Pour un URL Matchendirect de championnat donné,
    get_rankings renvoie un DataFrame correspondant au classement
    du championnat choisi."""
    championship_page = process_url(championship_url)
    championship_page = championship_page.find('table', {'id' : 'tableau_classement_lite'})
    count = 0
    all_rankings = []
    for elem in championship_page.findAll('tr'):
        if count != 1:
            classement = elem.find('th').get_text()
            count2 = 0
            for td in elem.findAll('td'):
                if count2 == 0:
                    equipe = td.get_text().strip()
                if count2 == 1:
                    points = td.get_text()
                if count2 == 2:
                    matchs = td.get_text()
                if count2 == 3:
                    goalaverage = td.get_text()
                count2 += 1
            all_rankings.append([classement,
                                 equipe,
                                 points,
                                 matchs,
                                 goalaverage])
            ranking = pd.DataFrame(columns=['Classement',
                                            'Equipe',
                                            'Points',
                                            'Matchs joués',
                                            'Goalaverage'],
                                   data=all_rankings)
            ranking = ranking.set_index('Classement', drop=False)
        count += 1
    return (ranking)


# Création de notre app :


app = dash.Dash(__name__)

# Dash permet de gérer du HTML et du CSS directement depuis Python.

app.layout = html.Div(className= 'container',
  children= [
  html.Div(children=[
    html.Div(children=[
      html.Br(),
      html.Br(),
      html.Br(),
      html.Div(
        "Rentrez une équipe qui joue en direct pour avoir des informations sur son match !",
        id='info_choice',
        ),
      html.Br(),
      html.Br(),
      html.Br(),
      dcc.Dropdown(
        id='team_choice',

    # On ne met pas ici les options car celles-ci dépendent du choix de l'utilisateur sur 'search_choice'

        placeholder= 'Entrez votre équipe',
        value= '[L1] PSG'
        ),
      html.Br(),
      html.Div('Les équipes sont chargées... Faites votre choix !',
        id='info_msg',
        style={
        'display': 'none'
        }
        ),
      html.Br(),
      html.Br(),
      html.Br(),
      ],
    style=
    {
    'width': '215px',
    'display': 'flex',
    'flex-direction': 'column',
    'flex-wrap': 'wrap'
    }
    ),
    dcc.Graph(
      id='bet_accuracy_evolution',
      figure={
          'layout': {
            'shapes' : dict(
            type="line",
            x0=0,
            x1=130,
            y0=2,
            y1=2,
            line=dict(
            color="Red",
            width=4,
            dash="dashdot"
            )
            ),
          'title': 'Quand parier ?'
          }
      },
       style={
           'width': '1100px'
      }
      ),
    html.Div(id='show_match_evolution')
    ],
    style=
    {
    'flex-direction': 'row',
    'margin': '15px',
    'display': 'flex',
    'flex-wrap': 'wrap'
    }
  ),
  html.Div(children=[
    dash_table.DataTable(
    id='ranking_table',
    style_table={
    'height': '450px',
    'overflowY': 'auto',
    'margin': 'auto'
    },
    ),
    html.Div(children=[
      dash_table.DataTable(
      id='stat_table',
      style_table={
      'width': '900px',
      'overflowX': 'auto',
      'margin-left': '30px',
      'margin-top': '30px',
      'padding': '0px 30px 30px 0px'
      }
      ),
      dash_table.DataTable(
      id='odds_table',
      style_table={
      'width': '900px',
      'margin-left': '30px',
      'margin-top': '30px',
      'overflowX': 'auto',
      'padding': '0px 30px 30px 0px'
      })
      ],
      style={
      'flex-direction': 'column',
      }
      )
    ],
    style={
    'display': 'flex',
    'padding': '15px 30px 30px 30px'
    }
    ),
  html.Br(),
  html.Br(),
  html.Br(),
  dcc.Interval(
    id='interval_component',
    interval=60*1000,
    n_intervals=0
    )
  ]
  )

# Les fonctions 'callback' permettent de faire réagir un (ou plusieurs) Output(s) (qu'on désigne pas son (leur) id),
# en fonction d'une liste d'Input(s) (qu'on désigne aussi par son (leur) id). Le deuxième terme des Inputs et Outputs
# correspond à l'attribut de cet Input dont on regarde le changement / de cet Output sur lequel on agit :
# après avoir défini les Outputs et les Inputs, on définit une fonction, dont les attributs sont les Inputs,
# qui sera appliquée chaque fois que la valeur d'un de ces Inputs change, et dont les valeurs retournée
# sont les Outputs.

# Ce premier callback rafraîchit chaque minute les choix du Dropdown de choix d'équipe.

@app.callback(
  [
  dash.dependencies.Output('team_choice', 'options'),
  dash.dependencies.Output('info_msg', 'style'),
  ],
  [
  dash.dependencies.Input('interval_component', 'n_intervals')
  ]
  )
def dropdown_options(refresh):
    to_return1 = [
    {'label' : '[' + list(BET_URLS_DICT.keys())[list(BET_URLS_DICT.values()).index(url)] + '] '+ live_team,
    'value' : '[' + list(BET_URLS_DICT.keys())[list(BET_URLS_DICT.values()).index(url)] + '] '+ live_team}
    for url in BET_URLS_DICT.values() for live_team in get_live_teams(url)]
    to_return2 = {'display' : 'flex'}
    return (to_return1, to_return2)

# Selon le choix d'équipe, on change le classement affiché (et sa mise en forme conditionnelle pour mettre
# en évidence l'équipe choisie dans ce classement).

@app.callback(
    [dash.dependencies.Output('ranking_table', 'columns'),
     dash.dependencies.Output('ranking_table', 'data'),
     dash.dependencies.Output('ranking_table', 'style_data_conditional')],
    [dash.dependencies.Input('team_choice', 'value')]
)

# Ici, on va effectuer une recherche pour avoir la correspondance du nom de l'équipe sur Betclic (qu'on a)
# en nom de l'équipe sur Matchendirect (que l'on cherche). Malheureusement, on n'a pas d'ID commun pour faire
# le pont entre les deux donc on utilise le module fuzzywuzzy (qu'il faut installer avant, avec
# !pip install fuzzywuzzy) dont on utilise la méthode process et sa fonction extractOne qui permet de
# repérer, à partir d'une string S et d'une liste de string A, l'élément de A qui a le plus de similitude
# avec S et, logiquement, les noms des équipes se ressemblant assez sur les deux sites, cela suffit.


def chose_rank_championship(championship):
    if championship is not None:
        live_team = championship[championship.find(' ') + 1:].rstrip()
        championship = championship[1:championship.find(']')]
        opponent_team = 'Olympique Marseille'
        for try_game in fetch_bet_urls(BET_URLS_DICT[championship]):
            if live_team in get_game_teams(try_game):
                for try_team in get_game_teams(try_game):
                    if live_team != try_team:
                        opponent_team = try_team

# Seul problème (il en fallait un !), l'équipe allemande de 'Mayence' (nom Betclic) est enregistrée
# au nom de 'Mainz 05' sur Matchendirect, on renomme donc cela à la main car, sinon, la fonction
# extractOne confond avec le Bayern Munich.

        if live_team == 'Mayence':
            live_team = 'Mainz'
        if opponent_team == 'Mayence':
            opponent_team = 'Mainz'
        df = get_ranking(MATCHENDIRECT_URLS_DICT[championship])
        chosen_team = process.extractOne(live_team,df['Equipe'])[0]
        opponent_team = process.extractOne(opponent_team, df['Equipe'])[0]
        index = int(df.index[df['Equipe'] == chosen_team].tolist()[0]) - 1
        opponent_index = int(df.index[df['Equipe'] == opponent_team].tolist()[0]) - 1
        style_data_conditional = [
        {
        'if':{
        'row_index': index,
        },
        'backgroundColor': '#98FB98',
        },
        {
        'if':{
        'row_index': opponent_index,
        },
        'backgroundColor': '#FD7B7B',
        },
        ]
        return (
      [{'name': col, 'id': col} for col in df.columns],
      df.to_dict('records'),
      style_data_conditional
      )
    else:
        return(
      [{'name': col, 'id': col} for col in get_ranking(MATCHENDIRECT_URLS_DICT['L1']).columns],
      get_ranking(MATCHENDIRECT_URLS_DICT['L1']).to_dict('records'),
      {}
      )

@app.callback(
    [dash.dependencies.Output('odds_table', 'columns'),
     dash.dependencies.Output('odds_table', 'data'),
     dash.dependencies.Output('stat_table', 'columns'),
     dash.dependencies.Output('stat_table', 'data'),
     dash.dependencies.Output('show_match_evolution', 'children'),
     dash.dependencies.Output('bet_accuracy_evolution', 'figure'),
  ],
  [dash.dependencies.Input('team_choice', 'value'),
  dash.dependencies.Input('interval_component', 'n_intervals')
  ]
  )
def get_stat_df(live_team, refresh):
    championship = live_team[1:live_team.find(']')]
    live_team = live_team[live_team.find(' ') + 1:].rstrip()
    championship_url = MATCHENDIRECT_URLS_DICT[championship]
    to_return1 = ([], [])
    to_return2 = ([], [])
    to_return3 = ''
    to_return4 = {}
    global x_y
    x_y = [[], []]
    text_to_add = ''
    game_score = ''
    good_game = None
    game_name = ''
    whole_df = []
    for try_game in fetch_bet_urls(BET_URLS_DICT[championship]):
        if live_team in get_game_teams(try_game):
            good_game = try_game
            whole_df = get_odds(good_game)
            if whole_df.empty:
                whole_df.insert(loc=0, column=get_game_name(good_game), value='Côtes indisponibles', allow_duplicates=True)
                text_to_add = " On ne trouve pas les côtes du match en question !"
                to_return1 = (
              [{'name': col, 'id': col} for col in whole_df.columns],
              whole_df.to_dict('records')
              )
            else:
                whole_df.insert(loc=0, column=get_game_name(good_game), value='Côtes :', allow_duplicates=True)
                to_return1 = (
              [{'name': col, 'id': col} for col in whole_df.columns],
              whole_df.to_dict('records')
              )
                min_odd = whole_df.min(axis=1)
                print(min_odd.values)
    page = process_url(MATCHENDIRECT_URLS_DICT[championship] + str(datetime.date.today().isocalendar()[0]) + '-' + str(datetime.date.today().isocalendar()[1]) + '/')
    target_page = page.findAll('tr', {'class': 'sl'})
    url_list = []
    if len(target_page) > 1:
        for elem in target_page:
            url_list.append(elem.find('a', href=True)['href'])
            link = 'https://www.matchendirect.fr/' + process.extractOne(live_team, url_list)[0]
        try:
            whole_df2 = infos_game(link=link, to_csv=False)
        except:
            pass
        else:
            if whole_df2 is not None:
                [x_y[0], x_y[1], predicted_score] = update_graph(whole_df, whole_df2, previous_x=[x_y[0]], previous_y=[x_y[1]])
                info_game = whole_df2.to_csv('file')
                to_return4 = {
                    'data':[
                        {'x': x_y[0], 'y': x_y[0]}
                    ],
                    'layout': {
                        'title': f"Notre prédiction pour {game_name} : score de {predicted_score}. <br> C'est un bon moment pour parier si {x_y[0][-1].values[0]} > 1 !"
                    }
                }
                to_return2 = (
          [{'name': col, 'id': col} for col in whole_df2.columns],
          whole_df2.to_dict('records')
          )
                game_score = f" Le score est actuellement de {whole_df2.iloc[0]['Buts']} - {whole_df2.iloc[0]['Buts']} !\n"
    elif len(target_page) == 1:
        link = 'https://www.matchendirect.fr/' + page.find(('tr'), {'class': 'sl'}).find('a', href=True)['href']
        try:
            whole_df2 = infos_game(link=link, to_csv=False)
        except:
            pass
        else:
            if whole_df2 is not None:
                [x_y[0], x_y[1], predicted_score] = update_graph(whole_df, whole_df2, previous_x=[x_y[0]], previous_y=[x_y[1]])
                info_game = whole_df2.to_csv('file')
                to_return4 = {
                    'data':[
                        {'x': x_y[0], 'y': x_y[0]}
                    ],
                    'layout': {
                        'title': f"Notre prédiction pour {game_name} : score de {predicted_score}. <br> C'est un bon moment pour parier si {x_y[0][-1].values[0]} > 1 !"
                    }
                }
                whole_df2.insert(loc=0, column=get_game_name(good_game), value=[get_game_teams(good_game)[0], get_game_teams(good_game)[1]], allow_duplicates=True)
                to_return2 = (
          [{'name': col, 'id': col} for col in whole_df2.columns],
          whole_df2.to_dict('records'),
          )
                game_score = f" Le score est actuellement de {whole_df2.iloc[0]['Buts']} - {whole_df2.iloc[0]['Buts']} !\n"
    if good_game is not None:
        game_name = get_game_name(good_game)
        try:
            game_time = int(whole_df2.index.get_level_values("Minute").values[0][:-1])
        except:
            to_return3 = f"C'est la mi-temps du match {game_name} !\n" + game_score + " Regardez ce que recommande notre modèle..."
        else:
            to_return3 = f"C'est la {game_time}e minute du match {game_name} !\n" + game_score
            if game_time < 20:
                to_return3 += " Il est encore trop tôt pour prédire l'avenir..."
            elif game_time >= 89:
                to_return3 += " Il est trop tard pour aller parier sur Betclic !"
            else:
                to_return3 += " Regardez ce que recommande notre modèle..."
        to_return3 += text_to_add
        to_return1 = list(to_return1)
        to_return2 = list(to_return2)
    return(to_return1[0], to_return1[1], to_return2[0], to_return2[1], to_return3, to_return4)

def update_graph(odd_df, stat_df, previous_x=[], previous_y=[]):
    target_odd = 1
    if stat_df is not None:
        stat_df = stat_df.to_csv('stat')
        this_minute_score, this_minute_proba = predict(pd.read_csv('stat'), 'alldata')
        for proba in this_minute_proba:
            if proba == -float('-inf'):
                proba = 1
        predicted_score = f"{this_minute_score[0]} - {this_minute_score[1]}"
        if isinstance(odd_df, list):
            odd_df = pd.DataFrame(odd_df)
        if odd_df.empty is False:
            try:
                target_odd = odd_df[predicted_score]
            except:
                target_odd = odd_df.min(axis=1)
        new_y = this_minute_proba[0] * this_minute_proba[1] * target_odd
        previous_x.append(new_y)
        try:
            new_x = int(whole_df2.index.get_level_values("Minute").values[0][:-1])
        except:
            previous_y.append(45)
        else:
            previous_y.append(new_x)
        return [previous_x, previous_y, predicted_score]
    return([], [], '')

# Lancement de l'app sur serveur local

# if __name__ == '__main__':
#    app.run_server(debug=True)

# gather_data('alldata', False, False)

# ignore_files()
