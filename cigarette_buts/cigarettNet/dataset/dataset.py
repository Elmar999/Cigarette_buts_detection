import sqlite3
import config
from . import query

class Dataset():
    def __init__(self , db_path):

        def connect_CTIS(ctis_path):
            con = sqlite3.connect(ctis_path)
            con.row_factory = sqlite3.Row
            return con

        self.con = connect_CTIS(db_path)

    def execute(self, query, payload=None, con=None):
        try:
            cursor = con.cursor()
            if payload:
                cursor.execute(query, payload)
            else:
                cursor.execute(query)
            res = cursor.fetchall()
            con.commit()
            cursor.close()
            return [dict(row) for row in res]
        except Exception as e:
            con.rollback()
            raise e




    def add_Paths(self, image_path, mask_path, json_path, con):
        '''
        add image_path, mask_path, json_path without any bg_path to database paths table
        '''
        image_path = '"{}"'.format(image_path)
        mask_path = '"{}"'.format(mask_path)
        json_path = '"{}"'.format(json_path)
        # query = 'INSERT INTO ' + config.table_name +' VALUES(' +p1+','+p2+','+p3+')'
        query = f'INSERT INTO {config.table_name} VALUES({image_path}, {mask_path}, {json_path}, NULL)'
        self.execute(query, con=con)

    def add_bg_path(self, bg_path, con):
        bg_path = '"{}"'.format(bg_path)
        query = f'INSERT INTO {config.table_name} VALUES(NULL, NULL, NULL, {bg_path})'
        self.execute(query, con=con)        

    def load_paths(self, conn, limit, json_path=False):
        '''
        load image , mask , json paths from database.
        Args: conn(db_type), json_path(bool), limit(int) 
        '''
        def _load(cur):
            paths = []
            rows = cur.fetchall()
            for row in rows:
                lst = []
                for i in row:
                    lst.append(i)
                paths.append(tuple(lst))        
            return paths

        cur = conn.cursor()
        if json_path:
            db_query = query.query_all + 'Limit '+f'{limit}'
            cur.execute(db_query)
            paths = _load(cur)
        else:
            db_query = query.query_not_json + 'Limit '+f'{limit}' 
            cur.execute(db_query)
            paths = _load(cur)
        return paths
            
   