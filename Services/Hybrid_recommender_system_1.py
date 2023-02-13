import pandas as pd
import numpy as np
import os
import pickle

#========== DATAS ==========
from azure.storage.blob import BlobClient

# ============ Cosine Similarity For specific articles ================================================
from sklearn.metrics.pairwise import cosine_similarity
def create_cosine_similarities_for_specific_articleId(used_articles_vector_matrix, articleId):
    # Cosine similarity for specific article
    specific_article_vector = used_articles_vector_matrix.loc[ used_articles_vector_matrix.index==articleId, :]
    specific_article_vector_inverse_df = used_articles_vector_matrix.loc[ used_articles_vector_matrix.index!=articleId ]
    specific_article_cosine_similarities = cosine_similarity(specific_article_vector_inverse_df, 
                                                          specific_article_vector)
    df_cosine_similarities_for_specific_article = pd.DataFrame( specific_article_cosine_similarities,
                                                       index = specific_article_vector_inverse_df.index,
                                                       columns = [articleId]).round(4)
    specific_article_similarities_scores = list(zip( df_cosine_similarities_for_specific_article.index,  
                                                     df_cosine_similarities_for_specific_article[articleId] ) )
    specific_article_similarities_scores = sorted(specific_article_similarities_scores, key=lambda x: x[1], 
                                               reverse=True)
    
    return specific_article_similarities_scores

# FInd Categories recommandation
def findCategoRecomm_for_nCat(top_3_prefered_categories, userId):
    recom = []
    query = top_3_prefered_categories[userId]
    for uid, user_ratings in query:
        recom.append(uid)
    return recom

# Recommend 'num' aricles for num specified for users without history
def pcaRecommend_n_articles(used_articles_vector_matrix, article_id, num):
    print('Recommending  '+ str(num) + " products similar to article Id " + str(article_id) + "...")
    print("-"*45)
    specific_article_similarities_scores = create_cosine_similarities_for_specific_articleId(used_articles_vector_matrix, article_id)
    result_list = []
    for result in specific_article_similarities_scores[0:num] :
        print('     Recommended article Id: ' + str(result[0]) + " (similarity score :" + str(result[1]) + ")") 
        result_list.append( "     Recommended article Id: " + str(result[0]) + " (similarity score :" + str(result[1]) + ")" )
    return result_list

# Get Predictions from 3 most liked Categories and Article similarities for users with history
def pcaPredictFromCatRatingsAndArticlesSimilarities(top_3_prefered_categories, userId, 
                                                 articles_metadata_for_used_articles, 
                                                 _5_most_read_articles, 
                                                 used_articles_vector_matrix):
    # Select categories most liked with collaborative filtering
    #top_n_prefered_categories = get_top_n_prefered_categories(predictions, 3)
    catRecom = findCategoRecomm_for_nCat(top_3_prefered_categories, userId)
    # Select 1 random article for each recommanded cat
    random_article_by_cat = [articles_metadata_for_used_articles[articles_metadata_for_used_articles['category_id'] == x]['article_id'].sample(1).values for x in catRecom]
    # For each category, select 5 articles by similarity with one random selected
    print(f'''Recommendations for user {userId}
Find below recommended articles based on similarities with read articles for most liked categories :''' )
    print('='*115)
    if len(catRecom) == 0 :
        print('You have no interaction for the moment')
        return {
            'You have no interaction for the moment'
            'We can recommend 5 articles from 5 differents categories that are the most read' : _5_most_read_articles

            }
            
    dict_result_article_by_category = {}

    if len(catRecom) == 1 :
        article_id = random_article_by_cat[0][0]
        cat_id = catRecom[0]
        print(f'For category {cat_id}')
        result_list = pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 5)
        dict_result_article_by_category['Category_1'] = cat_id
        dict_result_article_by_category[f'Recommended similar articles to {article_id}:'] = result_list
        print('='*75)
        return {
            'For Category_1 :' : cat_id,
            f'Recommended similar articles to {article_id}:' : result_list
        }
    
    if len(catRecom) == 2 :
        for i in range(0,2) :
            article_id = random_article_by_cat[i][0]
            cat_id = catRecom[i]
            print(f'For category {cat_id}')
            result_list = pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 3-i)
            dict_result_article_by_category['Category_'+str(i+1)] = cat_id
            dict_result_article_by_category[f'Recommended similar articles to {article_id} in category_'+str(i+1)+' :'] = result_list
            print('='*75)
        return {
                'For Category_1 :' : dict_result_article_by_category['Category_1'],
                f'Recommended similar articles to {random_article_by_cat[0][0]}:' : pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 3),
                'For Category_2 :' : dict_result_article_by_category['Category_2'],
                f'Recommended similar articles to {random_article_by_cat[1][0]}:' : pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 2)

            }
    
    if len(catRecom) == 3 :
        for i in range(0,3) :
            article_id = random_article_by_cat[i][0]
            cat_id = catRecom[i]
            print(f'For category {cat_id}')
            result_list = pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 3-i)
            dict_result_article_by_category['Category_'+str(i+1)] = cat_id
            dict_result_article_by_category[f'Recommended similar articles to {article_id} in category_'+str(i+1)+' :'] = result_list
            print('='*75)
        return {
                'For Category_1 :' : dict_result_article_by_category['Category_1'],
                f'Recommended similar articles to {random_article_by_cat[0][0]}:' : pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 3),
                'For Category_2 :' : dict_result_article_by_category['Category_2'],
                f'Recommended similar articles to {random_article_by_cat[1][0]}:' : pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 2),
                'For Category_3 :' : dict_result_article_by_category['Category_3'],
                f'Recommended similar articles to {random_article_by_cat[2][0]}:' : pcaRecommend_n_articles(used_articles_vector_matrix, article_id, 1)

            }

