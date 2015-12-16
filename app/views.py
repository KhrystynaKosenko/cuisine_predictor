import pandas as pd
import random
from flask import render_template
from flask import redirect
from flask import request
from flask import url_for
from app import app

from models_main import main_train
from models_main import make_json
from .forms import IngredientsArea

# create 'data' object with training data
# and train classifiers == 'clf'
data, clf = main_train()
data.get(source='test')
data.ingredients_for_random = data.test_data
random_ing = data.ingredients_for_random

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = IngredientsArea()
    if form.validate_on_submit():
        if 'submit' in request.form:
            body = form.body.data
            form.body.data = ''
            print body
            if body:    
                data.test_data = pd.read_json(make_json(body))
                print data.test_data
                data.lemmatize_ingredients(data.test_data, source='test')
                matrix_test = data.vectorize_ingredients(data.test_data)
                del data.test_data
                clf.predictions(matrix_test)
                results_list = zip(clf.models, clf.pred)
                return render_template('results.html', results=results_list)
        elif 'random' in request.form:
            index = random.randint(1, random_ing.shape[0])
            form.body.data =', '.join(random_ing['ingredients'][index])
            return render_template('index.html', form=form)
    elif 'random' in request.form:
        index = random.randint(1, random_ing.shape[0])
        form.body.data =', '.join(random_ing['ingredients'][index])
        return render_template('index.html', form=form)

    return render_template('index.html', form=form)
