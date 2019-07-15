import matplotlib.pyplot as plt
import json, requests
import pandas as pd
import numpy as np
import joblib
import base64
import io
from flask import Flask, render_template, url_for, redirect, request, abort

app = Flask(__name__)

pokemon = pd.read_csv('pokemon.csv')
combats = pd.read_csv('combats.csv')
pokemon['total'] = pokemon['HP'] + pokemon['Attack'] + pokemon['Defense'] + pokemon['Sp. Atk'] + pokemon['Sp. Def'] + pokemon['Speed']

@app.route('/')
@app.route('/home')
def home():
        return render_template('home.html')

@app.route('/hasil', methods=['POST','GET'])
def hasil():
    pokemon1 = request.form['pokemon1'].lower()
    pokemon2 = request.form['pokemon2'].lower()

    url1 = 'https://pokeapi.co/api/v2/pokemon/' + pokemon1
    url2 = 'https://pokeapi.co/api/v2/pokemon/' + pokemon2

    data1 = requests.get(url1)
    data2 = requests.get(url2)

    if str(data1) == '<Response [404]>':
        return render_template('error.html')
    elif str(data2)=='<Response [404]>':
        return render_template('error.html')

    filedata1 = data1.json()['forms']
    filedata2 = data2.json()['forms']
    
    name1 = filedata1[0]['name'].capitalize()
    name2 = filedata2[0]['name'].capitalize()

    picture1 = data1.json()['sprites']['front_default']
    picture2 = data2.json()['sprites']['front_default']

    # Prediction
    if pokemon1.capitalize() in pokemon['Name'].values and pokemon2.capitalize() in pokemon['Name'].values:
        firstPokemon = pokemon[pokemon['Name'] == pokemon1.capitalize()][['Name' ,'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'total']]
        secondPokemon = pokemon[pokemon['Name'] == pokemon2.capitalize()][['Name' ,'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'total']]
        battle = np.concatenate([firstPokemon.drop('Name', axis=1).values, secondPokemon.drop('Name', axis=1).values], axis=1)
        prediction = model.predict(battle)[0]
    
        compare = pd.concat([firstPokemon, secondPokemon])

        # PLOTTING
        plt.figure(figsize=(12,6))
        plt.subplot(161)
        plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['HP'], color=['blue', 'green'])
        plt.title('HP')

        plt.subplot(162)
        plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Attack'], color=['blue', 'green'])
        plt.title('Attack')

        plt.subplot(163)
        plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Defense'], color=['blue', 'green'])
        plt.title('Defense')

        plt.subplot(164)
        plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Sp. Atk'], color=['blue', 'green'])
        plt.title('Sp. Attack')

        plt.subplot(165)
        plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Sp. Def'], color=['blue', 'green'])
        plt.title('Sp. Defense')

        plt.subplot(166)
        plt.bar([compare.iloc[0]['Name'], compare.iloc[1]['Name']], compare['Speed'], color=['blue', 'green'])
        plt.title('Speed')

        plt.tight_layout()

        # GRAPH
        img = io.BytesIO()
        plt.savefig(img, format='png', transparent=True)
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        graph = 'data:image/png;base64,{}'.format(graph_url)

        # PROBABILITY
        if prediction == 1:
            prob = model.predict_proba(battle)[0][1] * 100
            win = name1
            result = {'prob':prob, 'win':win, 'graph':graph}
            return render_template('hasil.html', name1=name1, name2=name2, result=result, picture1=picture1, picture2 = picture2)
        else:
            prob = model.predict_proba(battle)[0][0] * 100
            win = name2
            result = {'prob':prob, 'win':win, 'graph':graph}
            return render_template('hasil.html', name1=name1, name2=name2, result=result, picture1=picture1, picture2 = picture2)
    else:
        abort(404)
    
@app.errorhandler(404)
def page_not_found(error):
	return render_template('error.html')

if __name__ == "__main__":
    model = joblib.load('modeljoblib')
    app.run(debug=True)