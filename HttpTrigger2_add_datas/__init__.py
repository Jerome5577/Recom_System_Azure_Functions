import logging
import json
import os

import azure.functions as func
from Services.Add_datas_to_list_file import add_datas
import tempfile


def main(req: func.HttpRequest, inputBlobInteractions: func.InputStream,
                                blobout: func.Out[bytes] ) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    userId = req.params.get('userId')
    articleId = req.params.get('articleId')
    if not userId or not articleId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            userId = req_body.get('userId')
            articleId = req_body.get('articleId')

    if userId and articleId:
        # ===== Open datas for user's Interactions
        json_user_new_interactions = json.load(inputBlobInteractions)
        # Call function to add datas
        data = add_datas(userId, articleId, json_user_new_interactions, blobout)

        return func.HttpResponse(json.dumps(data), headers={"content-type": "application/json"})
    else:
        return func.HttpResponse(
            '''
            This HTTP triggered function executed successfully.
            Pass a userId and an articleId in the query string or in the request body for a personalized response.
            '''
            ,
             status_code=200
        )
