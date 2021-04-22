from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('LoL.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    teamId = request.form['a']
    firstBlood = request.form['b']
    firstTower = request.form['c']
    firstInhib = request.form['d']
    firstBaron = request.form['e']
    firstDragon = request.form['f']
    firstRift = request.form['g']
    towerKills = request.form['h']
    inhibKills = request.form['i']
    baronKills = request.form['j']
    dragonKills = request.form['k']
    riftHeraldKills = request.form['l']
    arr = np.array([[teamId, firstBlood, firstTower, firstInhib, firstBaron, firstDragon, firstRift, towerKills, inhibKills, baronKills, dragonKills, riftHeraldKills]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)