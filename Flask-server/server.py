from flask import Flask
import pickle
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
Final = pickle.load(open("../model/Final.pkl","rb"))
latent_matrix_1 = pickle.load(open("../model/latent_matrix_1.pkl","rb"))
latent_matrix_2 = pickle.load(open("../model/latent_matrix_2.pkl","rb"))

app = Flask(__name__)
@app.route("/recommend/<MovieName>")
def recommend(MovieName):
    try:
        #take latent vector for selectred movie from both and collebrative matrices
        a_1 = np.array(latent_matrix_1.loc[MovieName.lower() + ' ']).reshape(1,-1)
        a_2 = np.array(latent_matrix_2.loc[MovieName.lower() + ' ']).reshape(1,-1)

        #calculate similarty of movies with other in list
        score_1 = cosine_similarity(latent_matrix_1, a_1).reshape(-1)
        score_2 = cosine_similarity(latent_matrix_2, a_2).reshape(-1)

        #an average measure of both content and collabrative
        hybrid = ((score_1+score_2)/2.0)

        #from data frame of similar movies
        dictDf = {'content':score_1,'collabrative':score_2,'hybrid':hybrid}
        similar = pd.DataFrame(dictDf, index = latent_matrix_1.index)

        #sort it on basis of content hybrid or collabrative
        similar.sort_values('hybrid',ascending=False,inplace=True)

        recommended_movie = similar[1:6].index.tolist()
        movie_list_id = []
        for m in recommended_movie:
            movie_list_id.append(int(Final[Final["title"] == m]["tmdbId"]))
        #     print(Final_1[Final_1["title"] == m])

        print(recommended_movie)
        print(movie_list_id)
    
        return movie_list_id
    except:
        return []        


    
if __name__ == "__main__":
    app.run(debug=True)
