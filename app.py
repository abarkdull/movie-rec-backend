import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

df = pd.read_csv('movie_data.csv')


@app.route('/')
def index():
    return jsonify({'message': "hello world"})


@app.route('/search')
def search():

    data_optimal = df.copy()
    movie = request.args.get('q')

    try:
        movie_id = data_optimal[data_optimal.Title == movie]['Movie_id'].values[0]
    except:
        return jsonify({'message': 'movie not found'})

    optimal_columns = ['Actors', 'Director', 'Description', 'Genre']

    data_optimal['important_features'] = get_important_columns(optimal_columns)
    cm_optimal = CountVectorizer().fit_transform(data_optimal['important_features'])
    cosine_similarity_matrix_optimal = cosine_similarity(cm_optimal)

    cs_scores_optimal = enumerate(cosine_similarity_matrix_optimal[movie_id])
    sorted_cs_scores_optimal = sorted(cs_scores_optimal, key = lambda x:x[1], reverse=True)

    movies_optimal = []
    for i in range(10):
        curr_id_optimal = sorted_cs_scores_optimal[i][0]
        movie_title_optimal = data_optimal[data_optimal.Movie_id == curr_id_optimal]['Title'].values[0]

        movies_optimal.append(movie_title_optimal)
    
    movies_optimal.pop(0)

    return jsonify({'movies': movies_optimal})


def get_important_columns(req_features):
    imp_columns = []
    for i in range(0, df.shape[0]):
        str_builder = ""
        for column in req_features:
            str_builder += str(df[column][i])
            str_builder += ' '
        imp_columns.append(str_builder)
    return imp_columns
    
if __name__ == '__main__':
    app.run(debug=True)