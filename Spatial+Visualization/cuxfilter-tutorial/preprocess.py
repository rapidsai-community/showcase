from pyproj import Proj, Transformer

def transform_coords(df, x='x', y='y'):
    transform_4326_to_3857 = Transformer.from_crs('epsg:4326', 'epsg:3857')
    df['x'], df['y'] = transform_4326_to_3857.transform(df[x].to_array(), df[y].to_array())
    return df

def process_trips(data):
    # Apply Transformation
    trips = transform_coords(data, x='latitude_start', y='longitude_start')


    # Note: days 0-4 are weekedays, days 5-6 are weekends 
    trips['day_type'] = 0
    trips.loc[trips.query('day>4').index, 'day_type'] = 1


    # Note: Data always has edge cases, such as the extra week anomalies of 2015 and 2016:
    # trips.groupby('year').week.max().to_pandas().to_dict() is {2014: 52, 2015: 53, 2016: 53, 2017: 52}
    # Since 2015 and 2016 have 53 weeks, we add 1 to global week count for their following years - 2016 & 2017
    # (data.year/2016).astype('int') => returns 1 if year>=2016, else 0
    year0 = int(trips.year.min()) #2014
    trips['all_time_week'] = data.week + 52*(data.year - year0) + (data.year/2016).astype('int')

    #Finally, we remove the unused columns and reorganize our dataframe:
    trips = trips[[
        'year', 'month', 'week', 'day', 'hour', 'gender', 'from_station_name',
        'from_station_id', 'to_station_id', 'x', 'y', 'from_station_name', 'to_station_name', 'all_time_week', 'day_type'
    ]]

    return trips