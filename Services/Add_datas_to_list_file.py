
import os, json


def add_datas(userId, articleId, user_new_interactions, blobout):
    interaction = {}
    interaction["userId"] = userId
    interaction['articleId'] = articleId

    user_new_interactions.append(interaction)
    
    datas = json.dumps(user_new_interactions, indent=2)
    blobout.set(datas)

    return {
      "Message": "New interaction added",
      "User": userId,
      'Article': articleId
    }