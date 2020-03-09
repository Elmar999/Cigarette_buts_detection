# from cigarettNet import config


query_all = ''' SELECT image_path,mask_path, json_path FROM paths '''
query_not_json = ''' SELECT image_path, mask_path FROM paths '''
query_bg = ''' Select bg_path FROM paths '''
