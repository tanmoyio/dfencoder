import pandas as pd
from dask import dataframe as dd

import os
import json


# _AZURE_RENAME_COLUMNS = {"location.countryOrRegion": "locationcountryOrRegion",
#                         "location.state": "locationstate",
#                         "location.city": "locationcity",
#                         "createdDateTime":"time",
#                         "deviceDetail.displayName":"deviceDetaildisplayName",
#                         "deviceDetail.browser":"deviceDetailbrowser",
#                         "deviceDetail.operatingSystem":"deviceDetailoperatingSystem",
#                         "status.failureReason":"statusfailureReason"}

# _AZURE_PARED_COLUMNS = ["userPrincipalName", 
#                     "appDisplayName", 
#                     "clientAppUsed", 
#                     "time", 
#                     "riskEventTypes_v2", 
#                     "locationcity", 
#                     "locationstate", 
#                     "locationcountryOrRegion", 
#                     "deviceDetaildisplayName", 
#                     "deviceDetailbrowser", 
#                     "deviceDetailoperatingSystem", 
#                     "statusfailureReason"]


def _explode_raw(df):
    df2 = pd.json_normalize(df['_raw'].apply(json.loads))
    return df2


def _azure_derived_features(df, timestamp_column, city_column, state_column, country_column, application_column):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf[timestamp_column])
    pdf['day'] = pdf['time'].dt.date
    pdf.sort_values(by=['time'], inplace=True)
    pdf.fillna("nan", inplace=True)
    pdf['overall_location'] = pdf[city_column] + ', ' + pdf[state_column] + ', ' + pdf[country_column]
    pdf['locincrement'] = pdf.groupby('day')['overall_location'].transform(lambda x: pd.factorize(x)[0] + 1)
    pdf['appincrement'] = pdf.groupby('day')[application_column].transform(lambda x: pd.factorize(x)[0] + 1)
    pdf["logcount"]=pdf.groupby('day').cumcount()
    pdf.drop('overall_location', inplace=True, axis = 1)
    return pdf


def _duo_derived_features(df, timestamp_column, city_column, state_column, country_column):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf[timestamp_column])
    pdf['day'] = pdf['time'].dt.date
    pdf.sort_values(by=['time'], inplace=True)
    pdf.fillna("nan", inplace=True)
    pdf['overall_location'] = pdf[city_column] + ', ' + pdf[state_column] + ', ' + pdf[country_column]
    pdf['locincrement'] = pdf.groupby('day')['overall_location'].transform(lambda x: pd.factorize(x)[0] + 1)
    pdf["logcount"]=pdf.groupby('day').cumcount()
    pdf.drop('overall_location', inplace=True, axis=1)
    return pdf


def _save_groups(df, outdir, source):
    df.to_csv(os.path.join(outdir, df.name.split('@')[0]+"_"+source+".csv"), index=False)
    return df


def proc_azure_logs(files, 
                    save_dir,
                    filetype = 'csv',
                    delimiter = ',',
                    groupby = 'userPrincipalName',
                    timestamp_column = 'createdDateTime',
                    city_column = 'location.city',
                    state_column = 'location.state',
                    country_column = 'location.countryOrRegion',
                    application_column = 'appDisplayName',
                    output_grouping = None,
                    extension=None,
                    min_records = 0):

    """
    Process Azure log files for DFP training.
    
    Parameters
    ----------
    files: str or List[str]
        A directory or filepath or list of filepaths
    save_dir: str
        The directory to save the training data
    filetype: str, default='csv'
        'csv' or 'json'
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

    if filetype == 'json':
        nested_logs = dd.read_json(files, lines=True)
        meta = pd.json_normalize(json.loads(nested_logs.head(1)['_raw'].to_list()[0])).iloc[:0,:].copy()
        azure_logs = nested_logs.map_partitions(lambda df: _explode_raw(df), meta=meta)
    else:
        azure_logs = dd.read_csv(files, delimiter=delimiter, dtype='object')

    azure_meta = {c: v for c, v in zip(azure_logs._meta, azure_logs._meta.dtypes)}
    azure_meta['time'] = 'datetime64[ns]'
    azure_meta['day'] = 'datetime64[ns]'
    azure_meta['locincrement'] = 'int'
    azure_meta['appincrement'] = 'int'
    azure_meta['logcount'] = 'int'

    azure_logs.persist()

    derived_azure = azure_logs.groupby(groupby).apply(lambda df: _azure_derived_features(df, timestamp_column, city_column, state_column, country_column, application_column), meta=azure_meta).reset_index(drop=True)

    if min_records > 0:
        user_entry_counts = azure_logs[[groupby, timestamp_column]].groupby(groupby).count().compute()
        trainees = [user for user, count in user_entry_counts.to_dict()[timestamp_column].items() if count > min_records]
        derived_azure[derived_azure[groupby].isin(trainees)].groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, "azure"), meta=derived_azure._meta).size.compute()
    else:
        derived_azure.groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, "azure"), meta=derived_azure._meta).size.compute()

    num_training_files = len([file for file in os.listdir(save_dir) if file.endswith('_azure.csv')])
    print("%i training files successfully created" % num_training_files)
    if num_training_files > 0:
        return True
    else:
        return False

def proc_duo_logs(files, 
                    save_dir,
                    delimiter = ',', 
                    groupby = 'username', 
                    timestamp_column = 'isotimestamp', 
                    city_column = 'location.city',
                    state_column = 'location.state',
                    country_column = 'location.country',
                    output_grouping = None, 
                    extension=None, 
                    min_records = 0):

    """
    Process Duo log files for DFP training.
    
    Parameters
    ----------
    files: str or List[str]
        A directory or filepath or list of filepaths
    save_dir: str
        The directory to save the training data
    filetype: str, default='csv'
        'csv' or 'json'
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

    duo_logs = dd.read_csv(files, delimiter=delimiter, dtype='object')

    duo_meta = {c: v for c, v in zip(duo_logs._meta, duo_logs._meta.dtypes)}
    duo_meta['time'] = 'datetime64[ns]'
    duo_meta['day'] = 'datetime64[ns]'
    duo_meta['locincrement'] = 'int'
    duo_meta['logcount'] = 'int'

    duo_logs.persist()

    derived_duo = duo_logs.groupby(groupby).apply(lambda df: _duo_derived_features(df, timestamp_column, city_column, state_column, country_column), meta=duo_meta).reset_index(drop=True)

    if min_records > 0:
        user_entry_counts = duo_logs[[groupby, timestamp_column]].groupby(groupby).count().compute()
        trainees = [user for user, count in user_entry_counts.to_dict()[timestamp_column].items() if count > min_records]
        derived_duo[derived_duo[groupby].isin(trainees)].groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, "duo"), meta=duo_meta).size.compute()
    else:
        derived_duo.groupby(output_grouping).apply(lambda df: _save_groups(df, save_dir, "duo"), meta=duo_meta).size.compute()

    num_training_files = len([file for file in os.listdir(save_dir) if file.endswith('_duo.csv')])
    print("%i training files successfully created" % num_training_files)
    if num_training_files > 0:
        return True
    else:
        return False
