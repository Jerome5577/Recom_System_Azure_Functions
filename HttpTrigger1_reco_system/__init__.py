import logging
import json

import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContainerClient

import os
import sys
import numpy as np
import pandas as pd
import pickle
import tempfile

from Services import Hybrid_recommender_system_1 as recom1

def loadInputBlobFile(inputBlobFile, temp_path, file_name):
    file_name = os.path.join(temp_path, file_name)
    with open(file_name, "w+b") as local_file:
        local_file.write(inputBlobFile.read())

    ext = file_name[-3:]

    if ext == "npy":
        payload = np.load(local_file.name)
    elif ext == "pkl":
        with open(local_file.name, "rb") as f:
            payload = pickle.load(f)
    elif ext == "csv":
        payload = pd.read_csv(local_file.name)

    return payload


def main(req: func.HttpRequest, inputBlobCat: func.InputStream,
                                inputBlobPCADfArticlesVectors: func.InputStream,
                                inputBlobArrayArticleIdsForPCADfArticlesVectors: func.InputStream,
                                inputBlobUsers5articles: func.InputStream,
                                inputBlobUsersClicks: func.InputStream,
                                inputBlobArticlesMetadata: func.InputStream ) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    userId = req.params.get('userId')
    if not userId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            userId = req_body.get('userId')

    if userId:
        #Global tempfile
        temp_path = tempfile.gettempdir()
        print("temp_path :", temp_path)

        # ===== Open datas for user's top_3_prefered_categories
        inputBlobFile = inputBlobCat
        file_name = 'top_3_prefered_categories.pkl'
        top_3_prefered_categories = loadInputBlobFile(inputBlobFile, temp_path, file_name)
        
        # ===== Open datas for cosine similarities computing
        #Vectors
        inputBlobFile = inputBlobPCADfArticlesVectors
        file_name = 'PCA_df_used_articles_vector_matrix.csv'
        used_articles_vector_matrix = loadInputBlobFile(inputBlobFile, temp_path, file_name)
        print('used_articles_vector_matrix :', used_articles_vector_matrix.shape )
        #Array indexes
        inputBlobFile = inputBlobArrayArticleIdsForPCADfArticlesVectors
        file_name = 'PCA_df_used_articles_vector_matrix_ArrayArticleIds.npy'
        datas_index = loadInputBlobFile(inputBlobFile, temp_path, file_name)
        ### Index = article_id
        used_articles_vector_matrix.index = datas_index
        print('used_articles_vector_matrix : ', used_articles_vector_matrix.shape)

        # ===== Open datas for users clicks
        inputBlobFile = inputBlobUsersClicks
        file_name = 'users_clicks_for_articles_and_category.csv'
        df_articleId_and_catId = loadInputBlobFile(inputBlobFile, temp_path, file_name)
        print('df_articleId_and_catId :', df_articleId_and_catId.shape)

        # ===== Open datas for articles_metadata
        inputBlobFile = inputBlobArticlesMetadata
        file_name = 'articles_metadata_for_used_articles.csv'
        articles_metadata_for_used_articles = loadInputBlobFile(inputBlobFile, temp_path, file_name)
        
        # ===== Open datas for 5 articles for users without history
        inputBlobFile = inputBlobUsers5articles
        file_name = '5_most_read_articles.npy'
        _5_most_read_articles = loadInputBlobFile(inputBlobFile, temp_path, file_name)

         # Make recommandations                                                                        
        recom = recom1.pcaPredictFromCatRatingsAndArticlesSimilarities(top_3_prefered_categories, 
                                                                    int(userId), 
                                                                    articles_metadata_for_used_articles,
                                                                    _5_most_read_articles,
                                                                    used_articles_vector_matrix)

        return func.HttpResponse(json.dumps(recom), headers={"content-type": "application/json"})

    else:
        return func.HttpResponse(
            '''
            This HTTP triggered function executed successfully.
            Pass a userId and an articleId in the query string or in the request body for a personalized response.
            '''
            ,
             status_code=200
        )

