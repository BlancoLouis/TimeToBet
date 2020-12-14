from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import numpy as np
import os

path_to_web_driver = "chromedriver"

stats_cats = ['Possession', 'Attaques', 'Attaques dangereuses',
              'Coups francs', 'Coups de pied arrêtés',
              'Buts', 'Tirs cadrés', 'Tirs non cadrés',
              'Tirs arrêtés', 'Tirs sur le poteau',
              'Pénaltys', 'Touches', 'Corners',
              'Hors-jeu', 'Fautes', 'Carton jaune',
              'Carton rouge', 'Remplacements']

championships_links = {'L1':
                       'https://www.matchendirect.fr/france/ligue-1',
                       'L2':
                       'https://www.matchendirect.fr/france/ligue-2',
                       'Liga':
                       'https://www.matchendirect.fr/espagne/primera-division/',
                       'SerieA':
                       'https://www.matchendirect.fr/italie/serie-a/',
                       'Bundelisga':
                       'https://www.matchendirect.fr/allemagne/bundesliga-1/'}


def select_champ(other_champs=None, champs=None):
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
                champs_links[elem] = championships_links[elem]
    else:
        for elem in championships_links.keys():
            champs_links[elem] = championships_links[elem]
    count_no_match = 0 # compte le nombre de championnats qui ont au moins un match en cours
    nb_champs_with_stats = 0 # ajoute à l'info précédente, la présence de statistiques pour les matchs
    for champ in champs_links.keys(): # on ouvre tous les championnats un par un 
        chrome_options = webdriver.ChromeOptions()
        browser = webdriver.Chrome(executable_path=path_to_web_driver,
                                   options=chrome_options)
        browser.get(champs_links[champ])
        time.sleep(2)
        try:
            browser.find_element_by_id('onetrust-accept-btn-handler').click()
        except:
            pass
        games_list = browser.find_elements_by_class_name('sl') # trouve les matchs en cours
        if len(games_list) > 0: # on vérifie qu'il y a bien des matchs en cours dans le championnat
            count_no_match += 1
            select_game(len(games_list), browser)
        browser.quit()

    if count_no_match == 0: # s'il n'y a pas de match alors le programme va attendre avant de redémarrer
        time.sleep(30)
        browser.quit()
        return
    return


def select_game(size, browser_origin):  # sélectionne tous les matchs en cours d'un championnat et agrège leurs stats une fois récupérées
    browser = browser_origin
    agg_stats = None
    for i in range(size):
        games_list = browser.find_elements_by_class_name('sl')
        if i < size: # si la liste des matchs en cours diminue pendant qu'on boucle
            link = games_list[i].find_element_by_css_selector('a').get_attribute('href')
            infos_game(link)
    browser.quit()
    return


def infos_game(link=None, to_csv=True):  # récupère les statistiques d'un match donné

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
        game_stats = pd.DataFrame(data=[list(t_1.values()), list(t_2.values())],
                                  index=index, columns=stats_cats)
        browser.quit()
    else:
        game_stats = None
    if game_stats is not None and to_csv:
        csv_file_name = str(team_1) + "_" + str(team_2) + '_' + str(minute)
        game_stats.to_csv(csv_file_name)
    return(game_stats)


def get_stats(other_champs=None, champs=None): # lance la boucle sur les championnats, récupère un dataframe de toutes les stats et en fait un fichier csv unique
    start_time = time.time()
    select_champ(other_champs, champs)
    end_time = time.time()
    duration = end_time - start_time
    if duration < 60: # rien ne sert plusieurs fois les stats d'un même match sur une même minute
        time.sleep(60 - duration)
    get_stats(other_champs, champs)


def red_card():
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

get_stats({'Pays-bas': 'https://www.matchendirect.fr/pays-bas/eredivisie/', 'Turquie': 'https://www.matchendirect.fr/turquie/bank-asya-1-lig/'})

# infos_game('https://www.matchendir    ect.fr/live-score/nacional-asuncion-river-plate-2198279.html')

# red_card()
