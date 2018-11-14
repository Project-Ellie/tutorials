def create_feature_columns():
    
    import numpy as np
    
    from tensorflow.feature_column import numeric_column as num
    from tensorflow.feature_column import bucketized_column as buck
    from tensorflow.feature_column import crossed_column as cross
    from tensorflow.feature_column import embedding_column as emb
    from tensorflow.feature_column import categorical_column_with_identity as cid
    
    ################################################################
    #  Numerical columns for the pre-processed features
    ################################################################
    feature_columns = [
        num(col) for col in [
            'DEP_DELAY',  
            'MEAN_TEMP_DEP','MEAN_VIS_DEP','WND_SPD_DEP',
            'MEAN_TEMP_ARR','MEAN_VIS_ARR','WND_SPD_ARR',
            'DIFF_LAT','DIFF_LON','DISTANCE']]
    
    ################################################################
    #  Crossed and embedded
    ################################################################
    lat_boundaries = np.arange(10,80,5).tolist()
    lon_boundaries = np.arange(-100, -55, 5).tolist()    
    cross_size = len(lat_boundaries) * len(lon_boundaries)

    arr_geo_emb = emb(cross([
        buck(num('ARR_LAT'), lat_boundaries), 
        buck(num('ARR_LON'), lon_boundaries)], cross_size), 10)

    dep_geo_emb = emb(cross([
        buck(num("DEP_LAT"), lat_boundaries), 
        buck(num("DEP_LON"), lon_boundaries)], cross_size), 10)

    dep_how_emb = emb(cross([
        cid("DEP_HOD", num_buckets=24), 
        cid("DEP_DOW", num_buckets=8)], 7*24), 10)

    ################################################################
    #  Crossed and embedded
    ################################################################
    return feature_columns + [dep_how_emb, arr_geo_emb, dep_geo_emb]
