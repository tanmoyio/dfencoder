import pandas as pd
import dask_cudf
import dask

import os
import glob
import json


_AZURE_RENAME_COLUMNS = {"location.countryorRegion": "locationcountryOrRegion",
                        "location.state": "locationstate",
                        "location.city": "locationcity",
                        "createdDateTime":"time",
                        "deviceDetail.displayName":"deviceDetaildisplayName",
                        "deviceDetail.browser":"deviceDetailbrowser",
                        "deviceDetail.operatingSystem":"deviceDetailoperatingSystem",
                        "status.failureReason":"statusfailureReason"}

_AZURE_PARED_COLUMNS = ["userPrincipalName", 
                    "appDisplayName", 
                    "clientAppUsed", 
                    "time", 
                    "riskEventTypes_v2", 
                    "locationcity", 
                    "locationstate", 
                    "locationcountryOrRegion", 
                    "deviceDetaildisplayName", 
                    "deviceDetailbrowser", 
                    "deviceDetailoperatingSystem", 
                    "statusfailureReason"]


def _explode_raw(df):
    df2 = pd.json_normalize(df['_raw'].apply(json.loads))
    return df2


def _save_groups(df, outdir):
    df.to_csv(os.path.join(outdir, df.name[:-11]+"_azure.csv"), index=False)
    return df


def proc_azure_logs(files, groupby_outdir, groupby = 'userPrincipalName', extension=None, min_records = 299):
    if isinstance(files, str):
        if os.path.isdir(files):
            if extension is not None:
                files = [file for file in os.listdir(files) if file.endswith(extension)]
            else:
                files = [file for file in os.listdir(files)]
        elif os.path.isfile(files):
            files = [files]
        else:
            files = []
    assert isinstance(files, list) and len(files) > 0, 'Please pass a directory, a file-path, or a list of file-paths containing the files to be processed'

    azure_logs = dask.dataframe.read_json(files, lines=True)

    meta = pd.json_normalize(json.loads(azure_logs.head(1)['_raw'].to_list()[0])).iloc[:0,:].copy()
    
    full_raw = azure_logs.map_partitions(lambda df: _explode_raw(df), meta=meta).rename(columns=_AZURE_RENAME_COLUMNS)
    pared_raw = full_raw[_AZURE_PARED_COLUMNS]

    user_entry_counts = pared_raw[[groupby, 'time']].groupby(groupby).count().compute()
    trainees = [user for user, count in user_entry_counts.to_dict()['time'].items() if count > min_records]

    pared_raw[pared_raw['userPrincipalName'].isin(trainees)].groupby('userPrincipalName').apply(lambda df: _save_groups(df, groupby_outdir), meta=pared_raw._meta).compute()
    
