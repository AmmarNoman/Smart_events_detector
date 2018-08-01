import pandas as pd
from Mapping import MappingData


from elasticsearch import Elasticsearch
import redis


class GetLocalData:
   

    # ====================================================================== #
    map_data = None
    
    def __init__(self, path):
        try:
            self.map_data = self.data(path)
        except Exception, e:
            print "Failed to load data\n", e
    

    def data(self, path):
        
        md = MappingData(path)
        return md.MappedData



class GetData:
   

    # -------------------- Redis constants -------------------------------------- #
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    # ----------------- Setting up tools  --------------------------------------- #
    red = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB
    )
        
    # ====================================================================== #
    map_data = None
    
    def __init__(self):
        try:
            self.map_data = self.data()
        except Exception, e:
            print "elasticsearch :: Failed to establish a new connection\n", e
    
        

    

    def data(self):
        index = "blindmotion"
        type = "dataset"
        size = 10000
        es = Elasticsearch([{
            'host': 'localhost',
            'port': 9200
        }])
        es.cluster.health(wait_for_status='yellow', request_timeout=1)
        if es.indices.exists(index=index):
            body = {
                
                "query": {
                    "match_all": {}
                }
            }
            res = es.search(index=index, doc_type=type, body=body)
            data_total = res['hits']['total']

            if data_total > 0:
                res = es.search(index=index, doc_type=type, body=body, size=data_total)
                res_data = res['hits']['hits']
                data_source = []
                for ind in xrange(len(res_data)):
                    data_source.append(res_data[ind]['_source'])

                DATA_LINK = pd.DataFrame(data=data_source)
                #raise


        
        return DATA_LINK

