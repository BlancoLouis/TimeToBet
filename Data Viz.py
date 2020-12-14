import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import json
import re
import datetime
import time
from urllib import request
import bs4
from selenium import webdriver
from pandas.util.testing import assert_frame_equal
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import threading


BET_URLS_DICT = {
    'L1': 'https://www.betclic.com/fr/paris-sportifs/football-s1/ligue-1-uber-eats-c4',
    'L2': 'https://www.betclic.com/fr/paris-sportifs/football-s1/ligue-2-bkt-c19',
    'PL': 'https://www.betclic.com/fr/paris-sportifs/football-s1/angl-premier-league-c3',
    'Liga': 'https://www.betclic.com/fr/paris-sportifs/football-s1/espagne-liga-primera-c7',
    'SerieA': 'https://www.betclic.com/fr/paris-sportifs/football-s1/italie-serie-a-c6',
    'Bundesliga': 'https://www.betclic.com/fr/paris-sportifs/football-s1/allemagne-bundesliga-c5',
    'LDC': 'https://www.betclic.com/fr/paris-sportifs/football-s1/ligue-des-champions-c8',
    'Europa': 'https://www.betclic.com/fr/paris-sportifs/football-s1/ligue-europa-c3453',
    'Grèce': 'https://www.betclic.com/fr/paris-sportifs/football-s1/grece-superleague-c38'
}

MATCHENDIRECT_URLS_DICT = {
    'L1': 'https://www.matchendirect.fr/france/ligue-1/',
    'L2': 'https://www.matchendirect.fr/france/ligue-2/',
    'PL': 'https://www.matchendirect.fr/angleterre/barclays-premiership-premier-league/',
    'Liga': 'https://www.matchendirect.fr/espagne/primera-division/',
    'SerieA': 'https://www.matchendirect.fr/italie/serie-a/',
    'Bundesliga': 'https://www.matchendirect.fr/allemagne/bundesliga-1/',
    'Grèce': 'https://www.matchendirect.fr/grece/super-league/'
}

BETCLIC_URL = 'https://www.betclic.com/fr/paris-sportifs/football-s1'

path_to_web_driver = 'chromedriver'

stats_cats = ['Possession', 'Attaques', 'Attaques dangereuses',
              'Coups francs', 'Coups de pied arrêtés',
              'Buts', 'Tirs cadrés', 'Tirs non cadrés',
              'Tirs arrêtés', 'Tirs sur le poteau',
              'Pénaltys', 'Touches', 'Corners',
              'Hors-jeu', 'Fautes', 'Carton jaune',
              'Carton rouge', 'Remplacements']

def process_url(url):
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    request_text = request.urlopen(req).read()
    bet_page = bs4.BeautifulSoup(request_text, 'lxml')
    return bet_page

# Pour l'url Betclic d'un championnat de football donné, fetch_bet_urls renvoie tous les URLs des matchs en direct

def fetch_bet_urls(championship_url):
    request_text = request.urlopen(championship_url).read()
    bet_page = bs4.BeautifulSoup(request_text, 'lxml')
    live_bet_game = []
    for elem in bet_page.findAll('app-live-event'):
        live_bet_game.append(elem.find('a').get('href'))
    live_bet_url = ['https://www.betclic.com' + elem for elem in live_bet_game]

# Cette étape semble peu inutile à première vue mais permet d'éviter un problème dans la situation où on boucle
# sur la liste d'URLs qu'on obtient via cette fonction et où on obtient un seul URL (un seul match en direct).
# Sans le 'if' qui arrive, notre code penserait que la liste d'URLs est composée des éléments 'h', 't', 't', 'p'...

    if len(live_bet_url) == 1:
        live_bet_url = [
            live_bet_url[0],
        ]
    return live_bet_url

# Cette fonction nous donne, pour un URL de match donné, les côtes de Score exact de ce match.
# Sur Betclic, la fonction qui gère ces côtes est régie par un code JS pour pouvoir être actualisée
# en continu.

def get_game_odd(game_url):
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

# Parfois (quand il vient d'y avoir un but par exemple), les côtes ne sont pas disponibles
# car le site les recalcule. Dans ce cas on récupère un dictionnaire vide.

    else:
      return {}


# Cette fonction nous renvoie une liste composée des noms des deux équipes (avec celle qui joue à domicile en 1er).

def get_game_teams(game_url):
    print(f'Fetching teams for game_url={game_url}')
    game_page = process_url(game_url)
    game = game_page.find('title').get_text()
    sep = game.find(' - ')
    end = game.find('|')
    team1, team2 = game[: sep], game[sep + 3 : end - 1]
    print(f'Found teams={[team1, team2]} for game_url={game_url}')
    return team1, team2

# On pourrait sûrement construire plus directement cette fonction à partir de la précédente.

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

# Cette fonction crée un DataFrame avec les côtes (score exact toujours) du match.


def get_odds(game_url):
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

# Cette fonction récupère les côtes d'un match chaque minute pendant n minutes, avec n = threshold,
# puis, une fois qu'il a fini, crée un fichier csv par match avec une ligne correspondant aux côtes
# pour une minute du match.
# file_nb correspond au nombre de fois où l'on veut répéter cette opération (qui dure threshold minutes),
# ainsi cette fonction doit durer threshold * file_nb minutes et crée un nombre file_nb de fichiers.
# Notons que le fichier file_nb = k + 1 contient toutes les lignes du fichier file_nb = k).

def create_odds_file(
    list_of_champ_links: list, threshold=4, count_file=0, file_dict={}, file_nb=0
):
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
            file_dict[game] = get_odds(game) # si file_dict[game] n'existe pas, on le crée
          else:
            file_dict[game] = pd.concat([file_dict[game], get_odds(game)]) # sinon on lui ajoute les côtes qu'on vient de trouver
    print(f'Done processing {len(list_of_champ_links)} elements.')
    count_file += 1
    if count_file <= threshold:
        end_time = time.time()
        duration = end_time - start_time

# Si on a fini de récupérer toutes les infos qui nous intéressent en moins d'une minute, on attend avant de continuer

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

#if __name__ == '__main__':
#    main()

def select_champ(other_champs=None, champs=None, count_file=0):
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
            select_game(len(games_list), browser, count_file)
        browser.quit()

    if count_no_match == 0: # s'il n'y a pas de match alors le programme va attendre avant de redémarrer
        time.sleep(30)
        browser.quit()
        return
    return


def select_game(size, browser_origin, count_file):  # sélectionne tous les matchs en cours d'un championnat et agrège leurs stats une fois récupérées
    browser = browser_origin
    agg_stats = None
    for i in range(size):
        games_list = browser.find_elements_by_class_name('sl')
        if i < size: # si la liste des matchs en cours diminue pendant qu'on boucle
            link = games_list[i].find_element_by_css_selector('a').get_attribute('href')
            infos_game(count_file, link)
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
        csv_file_name = str(team_1) + '_' + str(team_2) + '_' + str(minute)
        game_stats.to_csv(csv_file_name)
    return(game_stats)


def get_stats(other_champs=None, champs=None, count_file=0): # lance la boucle sur les championnats, récupère un dataframe de toutes les stats et en fait un fichier csv unique
    start_time = time.time()
    select_champ(other_champs, champs, count_file)
    end_time = time.time()
    duration = end_time - start_time
    if duration < 60: # rien ne sert plusieurs fois les stats d'un même match sur une même minute
        time.sleep(60 - duration)
    get_stats(other_champs, champs, count_file)


### DATA VISUALISATION

# On utilisera le module Dash, particulièrement utile pour créer des interactions entre les différents éléments
# qu'on organise sur le site créé. Les fonctions définies dans un premier temps servent de fonction intermédiaire
# pour récupérer nos données.


# Pour un URL Betclic de championnat donné, cette fonction retourne une liste d'équipes sous un format utile pour
# la partie Data Visualisation.

def get_live_teams(championship_url):
  list_live_teams = [live_team for game in fetch_bet_urls(championship_url) for live_team in get_game_teams(game)]
  return list_live_teams

# Pour un URL Matchendirect de championnat donné, get_rankings renvoie un DataFrame
# correspondant au classement du championnat choisi.

def get_ranking(championship_url):
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
      all_rankings.append([classement, equipe, points, matchs, goalaverage])
      ranking = pd.DataFrame(columns=['Classement', 'Equipe', 'Points', 'Matchs joués', 'Goalaverage'],
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
      dcc.RadioItems(
        options = [
        {'label' : 'Recherche par équipe', 'value' : 'by_team'},
        {'label' : 'Recherche par match en direct', 'value' : 'by_live_game'}
        ],
        value='by_live_game',
        id='search_choice',
        labelStyle={
        'display': 'flex',
        }
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
    'height': '500px',
    'overflowY': 'auto',
    'margin': 'auto'
    },
    ),
    html.Div(children=[
      dash_table.DataTable(
      id='stat_table',
      style_table={
      'width': '33%',
      'overflowX': 'auto',
      'margin': 'auto'
      }
      ),
      dash_table.DataTable(
      id='odds_table',
      style_table={
      'width': '33%',
      'overflowX': 'auto',
      'margin': 'auto'
      })
      ],
      style={
      'flex-direction': 'column',
      }
      )
    ],
    style={
    'display': 'flex',
    'margin': '10px'
    }
    ),
  html.Br(),
  html.Br(),
  html.Br(),
  ]
  )

# Les fonctions 'callback' permettent de faire réagir un (ou plusieurs) Output(s) (qu'on désigne pas son (leur) id),
# en fonction d'une liste d'Input(s) (qu'on désigne aussi par son (leur) id). Le deuxième terme des Inputs et Outputs
# correspond à l'attribut de cet Input dont on regarde le changement / de cet Output sur lequel on agit :
# après avoir défini les Outputs et les Inputs, on définit une fonction, dont les attributs sont les Inputs,
# qui sera appliquée chaque fois que la valeur d'un de ces Inputs change, et dont les valeurs retournée
# sont les Outputs.

# Selon le choix d'affichage de l'utlisateur (par équipe ou par match en direct), ce premier callback change
# les choix du Dropdown de choix.

@app.callback(
  [
  dash.dependencies.Output('team_choice', 'options'),
  dash.dependencies.Output('info_msg', 'style'),
  ],
  [
  dash.dependencies.Input('search_choice', 'value')
  ]
  )
def dropdown_options(value):
  if value == 'by_live_game':
    to_return1 = [
    {'label' : '[' + list(BET_URLS_DICT.keys())[list(BET_URLS_DICT.values()).index(url)] + '] '+ live_team,
    'value' : '[' + list(BET_URLS_DICT.keys())[list(BET_URLS_DICT.values()).index(url)] + '] '+ live_team}
    for url in BET_URLS_DICT.values() for live_team in get_live_teams(url)
    ]
    to_return2 = {'display' : 'flex'}
  else:
    to_return1 = [{'label' : 'PSG',
    'value' : 'PSG'}]
    to_return2 = {'display' : 'none'}
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
  [dash.dependencies.Output('stat_table', 'columns'),
  dash.dependencies.Output('stat_table', 'data'),
  dash.dependencies.Output('stat_table', 'style_data_conditional')],
  [dash.dependencies.Input('team_choice', 'value')]
  )
def get_stat_df(live_team):
  print(f'Fetching STATS for {live_team}')
  championship = live_team[1:live_team.find(']')]
  live_team = live_team[live_team.find(' ') + 1:].rstrip()
  championship_url = MATCHENDIRECT_URLS_DICT[championship]
  page = process_url(MATCHENDIRECT_URLS_DICT[championship] + str(datetime.date.today().isocalendar()[0]) + '-' + str(datetime.date.today().isocalendar()[1]) + '/')
  target_page = page.findAll('tr', {'class': 'sl'})
  if len(target_page) > 1:
    for elem in target_page:
      url_list.append(elem.find('a', href=True)['href'])
      link = 'https://www.matchendirect.fr/' + process.extractOne(live_team, url_list)
    whole_df = infos_game(link=link, to_csv=False)
    if whole_df is not None:
      return(
        [{'name': col, 'id': col} for col in whole_df.columns],
        whole_df.to_dict('records'),
        []
        )
    else:
      return([], [], [])
  elif len(target_page) == 1:
    link = 'https://www.matchendirect.fr/' + page.find(('tr'), {'class': 'sl'}).find('a', href=True)['href']
    whole_df = infos_game(link=link, to_csv=False)
    if whole_df is not None:
      return(
        [{'name': col, 'id': col} for col in whole_df.columns],
        whole_df.to_dict('records'),
        []
        )
    else:
      return([], [], [])
  else:
    return([], [], [])

@app.callback(
  [dash.dependencies.Output('odds_table', 'columns'),
  dash.dependencies.Output('odds_table', 'data')],
  [dash.dependencies.Input('team_choice', 'value')]
  )
def get_odds_df(live_game):
  championship = live_game[1:live_game.find(']')]
  live_game = live_game[live_game.find(' ') + 1:].rstrip()
  for try_game in fetch_bet_urls(BET_URLS_DICT[championship]):
    if live_game in get_game_teams(try_game):
      whole_df = get_odds(try_game)
      print(whole_df)
      if whole_df.empty:
        whole_df.insert(loc=0, column=get_game_name(try_game), value='Côtes indisponibles', allow_duplicates=True)
        return(
          [{'name': col, 'id': col} for col in whole_df.columns],
          whole_df.to_dict('records')
          )
      whole_df.insert(loc=0, column=get_game_name(try_game), value='Côtes :', allow_duplicates=True)
      print(whole_df)
      return(
        [{'name': col, 'id': col} for col in whole_df.columns],
        whole_df.to_dict('records')
        )
  return([], [])

def background():
  while True:
    global list_live_games
    debut = time.time()
    list_live_games = []
    for champ in BET_URLS_DICT.values():
      elem_to_add = get_live_teams(champ)
    for elem in elem_to_add:
      list_live_games.append(elem)
    fin = time.time()
    sleepy_time = 60 - (fin - debut)
    if sleepy_time > 0:
      print(f'Going to sleep for {sleepy_time}s')
      time.sleep(sleepy_time)


# Lancement de l'app sur serveur local

if __name__ == '__main__':
    app.run_server(debug = True)

'''
def start_server(app, **kwargs):
    def run():
        app.run_server(debug=True, **kwargs)

b = threading.Thread(name='background', target=background)
f = threading.Thread(name='foreground', target=foreground)

b.start()
f.start()
'''
