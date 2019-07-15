import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pokemon = pd.read_csv('pokemon.csv')
pokemon = pokemon.drop(['Type 1','Type 2','Generation','Legendary'],axis = 1)

combats = pd.read_csv('combats.csv')

pokemon['result'] = pokemon['HP'] + pokemon['Attack'] + pokemon['Defense'] + pokemon['Sp. Atk'] + pokemon['Sp. Def'] + pokemon['Speed']

name_dict = dict(zip(pokemon['#'], pokemon['Name']))
hp_dict = dict(zip(pokemon['#'], pokemon['HP']))
attack_dict = dict(zip(pokemon['#'], pokemon['Attack']))
defense_dict = dict(zip(pokemon['#'], pokemon['Defense']))
spattack_dict = dict(zip(pokemon['#'], pokemon['Sp. Atk']))
spdefense_dict = dict(zip(pokemon['#'], pokemon['Sp. Def']))
speed_dict = dict(zip(pokemon['#'], pokemon['Speed']))
result_dict = dict(zip(pokemon['#'], pokemon['result']))

dfcombats = combats.copy()

dfcombats['First_pokemon_name'] = dfcombats['First_pokemon'].replace(name_dict)
dfcombats['First_pokemon_hp'] = dfcombats['First_pokemon'].replace(hp_dict)
dfcombats['First_pokemon_attack'] = dfcombats['First_pokemon'].replace(attack_dict)
dfcombats['First_pokemon_defense'] = dfcombats['First_pokemon'].replace(defense_dict)
dfcombats['First_pokemon_spattack'] = dfcombats['First_pokemon'].replace(spattack_dict)
dfcombats['First_pokemon_spdefense'] = dfcombats['First_pokemon'].replace(spdefense_dict)
dfcombats['First_pokemon_speed'] = dfcombats['First_pokemon'].replace(speed_dict)
dfcombats['First_pokemon_result'] = dfcombats['First_pokemon'].replace(result_dict)

dfcombats['Second_pokemon_name'] = dfcombats['Second_pokemon'].replace(name_dict)
dfcombats['Second_pokemon_hp'] = dfcombats['Second_pokemon'].replace(hp_dict)
dfcombats['Second_pokemon_attack'] = dfcombats['Second_pokemon'].replace(attack_dict)
dfcombats['Second_pokemon_defense'] = dfcombats['Second_pokemon'].replace(defense_dict)
dfcombats['Second_pokemon_spattack'] = dfcombats['Second_pokemon'].replace(spattack_dict)
dfcombats['Second_pokemon_spdefense'] = dfcombats['Second_pokemon'].replace(spdefense_dict)
dfcombats['Second_pokemon_speed'] = dfcombats['Second_pokemon'].replace(speed_dict)
dfcombats['Second_pokemon_result'] = dfcombats['Second_pokemon'].replace(result_dict)

dfcombats['First_win'] = dfcombats.apply(
    lambda col: 1 if col['Winner'] == col['First_pokemon'] else 0, axis=1
)

x = dfcombats.drop(['First_pokemon', 'First_pokemon_name', 'Second_pokemon', 'Second_pokemon_name', 'Winner', 'First_win'], axis=1)
Y = dfcombats['First_win']

from sklearn.model_selection import train_test_split
xtr, yt, ytr, yt = train_test_split(
    x, 
    Y, 
    test_size=0.3, 
    random_state=101
)

from sklearn.linear_model import LogisticRegression
modelLog = LogisticRegression()
modelLog.fit(xtr, ytr)

import joblib
joblib.dump(modelLog,'modeljoblib')


