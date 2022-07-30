#!/bin/sh

FILES=$1
ORIGIN="azure"
SAVE_DIR="/home/nfs/sdavis/azure_test/20220730_script"
FILETYPE="csv"
DELIMITER="^"
GROUPBY="userPrincipalName"
TIMESTAMP="createdDateTime"
APP="appDisplayName"
CITY="location.city"
STATE="location.state"
COUNTRY="location.countryOrRegion"
MANAGER="m_name"
EXTENSION=".csv"
MIN_RECORDS=0

python preprocess.py --origin $ORIGIN \
 --files $FILES \
 --save_dir $SAVE_DIR \
 --filetype $FILETYPE \
 --delimiter $DELIMITER \
 --groupby $GROUPBY \
 --timestamp $TIMESTAMP \
 --app $APP \
 --manager $MANAGER \
 --extension $EXTENSION \
 --min_records $MIN_RECORDS
