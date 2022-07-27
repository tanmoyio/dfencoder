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


def _azure_derived_features(df):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf['time'])
    pdf['day'] = pdf['time'].dt.date
    pdf.sort_values(by=['time'])
    pdf.fillna("nan")
    pdf['locincrement'] = pdf.groupby('day')['locationcity'].transform(lambda x: pd.factorize(x)[0] + 1)
    pdf['appincrement'] = pdf.groupby('day')['appDisplayName'].transform(lambda x: pd.factorize(x)[0] + 1)
    pdf["logcount"]=pdf.groupby('day').cumcount()
    return pdf


def _duo_derived_features(df):
    pdf = df.copy()
    pdf['time'] = pd.to_datetime(pdf['time'])
    pdf['day'] = pdf['time'].dt.date
    pdf.sort_values(by=['time'])
    pdf.fillna("nan")
    pdf['locincrement'] = pdf.groupby('day')['locationcity'].transform(lambda x: pd.factorize(x)[0] + 1)
    pdf["logcount"]=pdf.groupby('day').cumcount()
    return pdf


def _save_groups(df, outdir):
    df.to_csv(os.path.join(outdir, df.name[:-11]+"_azure.csv"), index=False)
    return df


def proc_azure_logs(files, groupby_outdir, groupby = 'userPrincipalName', output_grouping = None, extension=None, min_records = 299):
    if output_grouping is None:
        output_grouping = groupby
    
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

    pared_meta = {c: v for c, v in zip(pared_raw._meta, pared_raw._meta.dtypes)}
    pared_meta['day'] = 'datetime64[ns]'
    pared_meta['time'] = 'datetime64[ns]'
    pared_meta['locincrement'] = 'int'
    pared_meta['appincrement'] = 'int'
    pared_meta['logcount'] = 'int'

    pared_raw.persist()

    derived_raw = pared_raw.groupby(groupby).apply(lambda df: _azure_derived_features(df), meta=pared_meta).reset_index(drop=True)

    user_entry_counts = pared_raw[[groupby, 'time']].groupby(groupby).count().compute()
    trainees = [user for user, count in user_entry_counts.to_dict()['time'].items() if count > min_records]

    derived_raw[derived_raw[groupby].isin(trainees)].groupby(output_grouping).apply(lambda df: _save_groups(df, groupby_outdir), meta=derived_raw._meta).size.compute()

def proc_duo_logs(files, groupby_outdir, groupby = 'username', output_grouping = None, extension=None, min_records = 299):

    if output_grouping is None:
        output_grouping = groupby
    
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

    duo_logs = dask.dataframe.read_csv(files)
    duo_cleaned = duo_logs.rename(mapper = lambda col: col.replace('[_,.,{,},:]',''))

    duo_meta = {c: v for c, v in zip(duo_cleaned._meta, duo_cleaned._meta.dtypes)}
    duo_meta['day'] = 'datetime64[ns]'
    duo_meta['time'] = 'datetime64[ns]'
    duo_meta['locincrement'] = 'int'
    duo_meta['logcount'] = 'int'

    duo_cleaned.persist()

    derived_duo = duo_cleaned.groupby(groupby).apply(lambda df: _duo_derived_features(df), meta=duo_meta).reset_index(drop=True)

    user_entry_counts = duo_cleaned[[groupby, 'time']].groupby(groupby).count().compute()
    trainees = [user for user, count in user_entry_counts.to_dict()['time'].items() if count > min_records]

    derived_duo[derived_duo[groupby].isin(trainees)].groupby(output_grouping).apply(lambda df: _save_groups(df, groupby_outdir), meta=derived_duo._meta).size.compute()

