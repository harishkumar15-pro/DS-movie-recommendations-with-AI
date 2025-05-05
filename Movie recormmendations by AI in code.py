
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Clean 'genres'
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Similarity Matrix
similarity = cosine_similarity(tfidf_matrix)

# Save plot folder
if not os.path.exists('static'):
    os.makedirs('static')

# Top 10 Most Rated
top_movies = ratings['movieId'].value_counts().head(10)
top_titles = movies.set_index('movieId').loc[top_movies.index]['title']

plt.figure(figsize=(10, 5))
top_movies.plot(kind='bar')
plt.xticks(ticks=range(10), labels=top_titles, rotation=45)
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Movie Title')
plt.ylabel('Number of Ratings')
plt.tight_layout()
plt.savefig('static/top_movies.png')
plt.close()

# Recommend Function
def recommend(title):
    if title not in movies['title'].values:
        return ["Movie not found."]
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations, error = [], ''
    if request.method == 'POST':
        movie_name = request.form['movie']
        recommendations = recommend(movie_name)
        if recommendations == ["Movie not found."]:
            error = "Movie not found. Please try another title."
            recommendations = []
    return render_template('index.html', recommendations=recommendations, image='static/top_movies.png', error=error)

if __name__ == '__main__':
    app.run(debug=True)

    movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance
5,Father of the Bride Part II (1995),Comedy
6,Heat (1995),Action|Crime|Thriller
7,Sabrina (1995),Comedy|Romance
8,Tom and Huck (1995),Adventure|Children
9,Sudden Death (1995),Action
10,GoldenEye (1995),Action|Adventure|Thriller

userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
1,6,4.0,964982224
1,47,5.0,964983815
1,50,5.0,964982931
2,1,5.0,964982931
2,2,3.0,964982224
2,3,4.0,964982703
2,5,2.0,964983815
2,7,5.0,964981247

<!DOCTYPE html>
<html>
<head>
    <title>Movie Recommender</title>
</head>
<body>
    <h1>AI Movie Recommendation System</h1>
    <form method="POST">
        <label for="movie">Enter a movie title:</label>
        <input type="text" name="movie" required>
        <button type="submit">Recommend</button>
    </form>

    {% if error %}
        <p style="color:red;">{{ error }}</p>
    {% endif %}

    {% if recommendations %}
        <h2>Recommended Movies:</h2>
        <ul>
            {% for movie in recommendations %}
                <li>{{ movie }}</li>
            {% endfor %}
        </ul>
    {% endif %}

    <img src="{{ image }}" alt="Top 10 Most Rated Movies">
</body>
</html>
