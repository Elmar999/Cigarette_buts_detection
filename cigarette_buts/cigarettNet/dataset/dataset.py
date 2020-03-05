import sqlite3
import config

class Dataset():
    def __init__(self , db_name):
        self.db_name = db_name
        self.db = MySQLdb.connect(db=config.db_name)



        
    def add_Path(image_path, mask_path, json_path):
        p1 = '"{}"'.format(image_path)
        p2 = '"{}"'.format(mask_path)
        p3 = '"{}"'.format(json_path)

        con.execute("INSERT INTO" + config.table_name +"VALUES(" +p1+','+p2+','+p3+')')
    
    def load_paths(image_path, mask_path, json_path=False):
        pass

        
    def execute(query, payload=None, con=None):
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


    # CTIS_MAIN_TABLE = "tile"
    def connect_CTIS(ctis_path):
        con = sqlite3.connect(ctis_path)
        con.row_factory = sqlite3.Row
        return con