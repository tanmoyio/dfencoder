import time
import datetime
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client
import numpy as np
import os
import sys
import argparse
import json


parser = argparse.ArgumentParser(description="Process Duo or Azure logs for DFP")
parser.add_argument('--origin', choices=['duo', 'azure'], default='duo', help='the type of logs to process: duo or azure')
parser.add_argument('--files', default=None, help='The directory containing the files to process')
parser.add_argument('--save_dir', default=None, help='The directory to save the processed files')
parser.add_argument('--filetype', default='csv', choices=['csv', 'json', 'jsonline'], help='Switch between csv and jsonlines for processing Azure logs')
parser.add_argument('--explode_raw', action='store_true', help='Option to explode the _raw key from a jsonline file')
parser.add_argument('--delimiter', default=',', help='The CSV delimiter in the files to be processed')
parser.add_argument('--groupby', default=None, help='The column to be aggregated over. Usually a username.')
parser.add_argument('--timestamp', default=None, help='The name of the column containing the timing info')
parser.add_argument('--city', default=None, help='The name of the column containing the city')
parser.add_argument('--state', default=None, help="the name of the column containing the state")
parser.add_argument('--country', default=None, help="The name of the column containing the country")
parser.add_argument('--app', default='appDisplayName', help="The name of the column containing the application. Does not apply to Duo logs.")
parser.add_argument('--manager', default=None, help='The column containing the manager name. Leave blank if you want user-level results')
parser.add_argument('--extension', default=None, help='The extensions of the files to be loaded. Only needed if there are other files in the directory containing the files to be processed')
parser.add_argument('--min_records', type=int, default=0, help='The minimum number of records needed for a processed user to be saved.')


_DEFAULT_DATE = '1970-01-01T00:00:00.000000+00:00'


def _if_dir_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def _explode_raw(df):
    df2 = pd.json_normalize(df['_raw'].apply(json.loads))
    return df2


def _derived_features(df, timestamp_column, city_column, state_column, country_column, application_column):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf[timestamp_column], errors='coerce')
    pdf['day'] = pdf['time'].dt.date
    pdf.fillna({'time': pd.to_datetime(_DEFAULT_DATE), 'day': pd.to_datetime(_DEFAULT_DATE).date()}, inplace = True)
    pdf.sort_values(by=['time'], inplace=True)
    overall_location_columns = [col for col in [city_column, state_column, country_column] if col is not None]
    if len(overall_location_columns) > 0:
        pdf['overall_location'] = pdf[overall_location_columns].apply(lambda x: ', '.join(x), axis=1)
        pdf['loc_cat'] = pdf.groupby('day')['overall_location'].transform(lambda x: pd.factorize(x)[0] + 1)
        pdf.fillna({'loc_cat': 1}, inplace = True)
        pdf['locincrement'] = pdf.groupby('day')['loc_cat'].expanding(1).max().droplevel(0)
        pdf.drop(['overall_location', 'loc_cat'], inplace=True, axis=1)
    if application_column is not None:
        pdf['app_cat'] = pdf.groupby('day')[application_column].transform(lambda x: pd.factorize(x)[0] + 1)
        pdf.fillna({'app_cat': 1}, inplace = True)
        pdf['appincrement'] = pdf.groupby('day')['app_cat'].expanding(1).max().droplevel(0)
        pdf.drop('app_cat', inplace=True, axis=1)
    pdf["logcount"]=pdf.groupby('day').cumcount()
    
    return pdf


def _save_groups(df, outdir, source):
    df.to_csv(os.path.join(outdir, df.name.split('@')[0]+"_"+source+".csv"), index=False)
    return df


def _parse_time(df, timestamp_column):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf[timestamp_column])
    pdf['day'] = pdf['time'].dt.date
    return pdf


def proc_logs(files, 
                save_dir,
                log_source = 'duo',
                filetype = 'csv',
                storage_options = {},
                explode_raw = False,
                delimiter = ',',
                groupby = 'userPrincipalName',
                timestamp_column = 'createdDateTime',
                city_column = None,
                state_column = None,
                country_column = None,
                application_column = None,
                output_grouping = None,
                extension=None,
                min_records = 0):
    """
    Process log files for DFP training.
    
    Parameters
    ----------
    files: str or List[str]
        A directory or filepath or list of filepaths
    save_dir: str
        The directory to save the training data
    log_source: str
        The source of the logs. Used primarily for tracing training data provenance.
    filetype: str, default='csv'
        'csv', 'json', or 'jsonline'
    storage_options: dict
        any arguments to pass to dask if trying to access data from remote locations such as AWS
    explode_raw: bool
        This indicates that the data is in a nested jsonlines object with the _raw key
    delimiter: str, default=','
        The csv delimiter
    groupby: str, default='userPrincipalName'
        The column name to aggregate over for derived feature creation.
    timestamp_column: str, default='createdDateTime
        The column name containing the timestamp
    city_column: str, default='location.city'
        The column name containing the city location data
    state_column: str, default='location.state'
        The column name containing the state location data
    country_column: str, default='location.countryOrRegion
        The column name containing the country location data
    application_column: str, default='appDisplayName'
        The column name containing the app name data
    output_grouping: str, optional
        The column to aggregate the output training data. If None, this defaults to the aggregation level specified in the groupby parameter.
        This is where you would specify the manager name column, if training is being done by manager group.
    extension: str, optional
        Specify the file extension to load, if the directory contains additional files that should not be loaded.
    min_records: int, default=0
        The minimum number of records that need to be observed to save the data for training. Setting this to 0 creates data for all users.
    
    Returns
    -------
    bool
        True if more than 1 training file is returned, else False is returned

    """
    if output_grouping is None:
        output_grouping = groupby

    _if_dir_not_exists(save_dir)
    
    if isinstance(files, str):
        if os.path.isdir(files):
            if extension is not None:
                files = [os.path.join(files, file) for file in os.listdir(files) if file.endswith(extension)]
            else:
                files = [os.path.join(files, file) for file in os.listdir(files)]
        elif os.path.isfile(files):
            files = [files]
        else:
            files = []
    assert isinstance(files, list) and len(files) > 0, 'Please pass a directory, a file-path, or a list of file-paths containing the files to be processed'

    start_time = time.perf_counter()
    
    if filetype == 'jsonline':
        if explode_raw:
            nested_logs = dd.read_json(files, lines=True, storage_options=storage_options)
            meta = pd.json_normalize(json.loads(nested_logs.head(1)['_raw'].to_list()[0])).iloc[:0,:].copy()
            logs = nested_logs.map_partitions(lambda df: _explode_raw(df), meta=meta).fillna('nan')
        else:
            logs = dd.read_json(files, lines=True, storage_options=storage_options).fillna('nan')
    elif filetype == 'json':
        logs = dd.read_json(files, storage_options=storage_options).fillna('nan')
    else:
        logs = dd.read_csv(files, delimiter=delimiter, storage_options=storage_options, dtype='object').fillna('nan')

    logs_meta = {c: v for c, v in zip(logs._meta, logs._meta.dtypes)}
    logs_meta['time'] = 'datetime64[ns]'
    logs_meta['day'] = 'datetime64[ns]'
    if city_column is not None or state_column is not None or country_column is not None:
        logs_meta['locincrement'] = 'int'
    if application_column is not None:
        logs_meta['appincrement'] = 'int'
    logs_meta['logcount'] = 'int'

    derived_logs = logs.groupby(groupby).apply(lambda df: _derived_features(df, timestamp_column, city_column, state_column, country_column, application_column), meta=logs_meta).reset_index(drop=True)

    if min_records > 0:
        logs = logs.persist()
        user_entry_counts = logs[[groupby, 'day']].groupby(groupby).count().compute()
        trainees = [user for user, count in user_entry_counts.to_dict()['day'].items() if count > min_records]
        derived_logs[derived_logs[groupby].isin(trainees)].groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, log_source), meta=derived_logs._meta).size.compute()
    else:
        derived_logs.groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, log_source), meta=derived_logs._meta).size.compute()

    timing = datetime.timedelta(seconds = time.perf_counter() - start_time)

    num_training_files = len([file for file in os.listdir(save_dir) if file.endswith('_{log_source}.csv'.format(log_source=log_source))])
    print("{num_files} training files successfully created in {time}".format(num_files=num_training_files, time=timing))
    if num_training_files > 0:
        return True
    else:
        return False


def _run():
    opt = parser.parse_args()

    client = Client()
    client.restart()

    print('Beginning {origin} pre-processing'.format(origin=opt.origin))
    proc_logs(files=opt.files, 
                        log_source=opt.origin,
                        save_dir=opt.save_dir,
                        filetype=opt.filetype, 
                        explode_raw=opt.explode_raw,
                        delimiter=opt.delimiter, 
                        groupby=opt.groupby or 'userPrincipalName',
                        timestamp_column=opt.timestamp or 'createdDateTime',
                        city_column=opt.city,
                        state_column=opt.state,
                        country_column=opt.country,
                        application_column=opt.app,
                        output_grouping=opt.manager,
                        extension=opt.extension,
                        min_records=opt.min_records)
    
    client.close()

if __name__ == '__main__':
    _run()