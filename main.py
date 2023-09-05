from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()


@app.get('/serdata/')
def userdata(user_id: str) -> dict:
    """
    Devuelve total gastado, porcentaje de recomendaciones y total de reseñas por usuario
    """
    df = pd.read_csv('endpoint1.csv')
    user_data = df[df['user_id'] == user_id]

    # Calcula total gastado por usuario
    total_spent = user_data['price'].sum()

    # Calcula porcentaje de recomendaciones y total de reviews
    total_recommended = user_data['recommend'].sum()
    total_reviews = user_data.shape[0]
    recommendation_percentage = (total_recommended / total_reviews) * 100

    return {
        'Total Gastado': total_spent,
        'Porcentaje de recomendaciones': recommendation_percentage,
        'Reseñas totales': total_reviews
    }


# 2


@app.get('/countreviews/')
def countreviews(fecha1: int, fecha2: int):
    df = pd.read_csv('endpoint1.csv')
    fil = df[(df.year >= fecha1) & (df.year <= fecha1)]
    return {
        'la cantidad de usuarios': fil.user_id.count(),
        'porcentaje de reviews': fil.recommend.count() / len(fil) * 100
    }


# 3


@app.get('/genero/')
def genero(genre: str) -> int:
    ''' Ingresa un genero y te devuelve el puesto en el que se encuentra en el ranking de generos'''
    df = pd.read_csv('endpoint3.csv')
    # Verifica si el genero ingresado existe en el DataFrame
    if genre not in df.columns:
        return "Genero Invalido"

    # Multiplica la columna del genero por la columna de playtime_forever
    genre_playtime = df[genre] * df['playtime_forever']

    # Calcula el total de horas jugadas en el genero
    total_genre_playtime = genre_playtime.sum()

    # Lista de generos pertenecientes al DataFrame
    genre_columns = [
        'Action', 'Indie', 'Adventure', 'Casual', 'Fighting', 'Multiplayer',
        'Puzzle', 'RPG', 'Sandbox', 'Shooter', 'Simulation', 'Singleplayer',
        'Sports', 'Strategy', 'Survival', 'Zombies'
    ]

    # Calcula el total de horas jugadas en cada genero
    total_playtimes = {
        genre: (df[genre] * df['playtime_forever']).sum()
        for genre in genre_columns
    }

    # Ordena los generos de mayor a menor
    sorted_genres = sorted(total_playtimes.items(),
                           key=lambda x: x[1],
                           reverse=True)

    # Busca el puesto del genero ingresado
    rank = 1
    for genre, playtime in sorted_genres:
        if genre == genre:
            return rank
        rank += 1


# 4


@app.get('/userforgenre/')
def userforgenre(genre: str) -> dict:
    ''' Ingresa un genero y te devuelve el top 5 de usuarios con más horas de juego en el genero dado, con su URL (del user) y user_id. '''
    df = pd.read_csv('endpoint4.csv')
    # Verifica si el genero ingresado existe en el DataFrame
    if genre not in df.columns:
        return "Invalid genre"

    # Filtra el DataFrame por el genero ingresado
    genre_df = df[df[genre] == 1]

    # Ordera el DataFrame por playtime_forever
    sorted_genre_df = genre_df.sort_values(by='playtime_forever',
                                           ascending=False)

    # Extrae los primeros 5 usuarios
    top_5 = sorted_genre_df.head(5)

    # Crea un diccionario con los usuarios y sus datos
    top_users_dict = {}
    for _, row in top_5.iterrows():
        top_users_dict[row['user_id']] = {
            'user_url': row['user_url_x'],
            'playtime_forever': row['playtime_forever']
        }

    return top_users_dict


#5
@app.get('/developer/')
def developer(year: str):
    df = pd.read_csv('endpoint5.csv')
    fil = df[df.year == year]
    return {'Año': year, 'Contenido Free': fil.price.to_list()[0] * 100}


#6
@app.get('/sentiment_analysis/')
def sentiment_analysis(year: str):
    df = pd.read_csv('endpoint6.csv')
    fil = df[df.year == year]
    return {
        'Año': year,
        'Positivo': fil.positivo.to_list()[0],
        'Negativo': fil.negativo.to_list()[0],
        'Neutral': fil.neutral.to_list()[0]
    }


# ML
@app.get('/recomendacion/')
def recomendacion(title: str) -> list:
    df = pd.read_csv('reco.csv')

    # Combinamos reviews por titulos

    df["review"] = df["review"].fillna("")
    grouped = df.groupby('item_name').agg(lambda x: ' '.join(x)).reset_index()

    # 2. Calcula matriz TF-IDF usando stop words en inglés
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(grouped['review'])

    # Calcula matriz de similaridad del coseno
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = grouped.index[grouped['item_name'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    item_indices = [i[0] for i in sim_scores]
    return grouped['item_name'].iloc[item_indices].tolist()
