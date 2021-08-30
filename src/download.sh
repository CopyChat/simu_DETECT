#!/bin/bash - 
#======================================================
#
#          FILE: jj.sh
# 
USAGE="./ERA5_download"
# 
#   DESCRIPTION: download ERA5 WRF forcing data @ pressure level
# 
#       OPTIONS: ---
#  REQUIREMENTS: --- GetERA5-sfc.py, and GetERA5-ml.py
#          BUGS: --- unknown
#         NOTES: --- run @CCuR
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
#  ORGANIZATION: 
#       CREATED: 15/11/2019 21:09
#      REVISION: 1.0
#=====================================================
set -o nounset           # Treat unset variables as an error
. ~/Code/Shell/functions.sh   # ctang's functions

while getopts ":tf:" opt; do
    case $opt in
        t) TEST=1 ;;
    f) file=$OPTARG;;
\?) echo $USAGE && exit 1
    esac
done

shift $(($OPTIND - 1))
#=====================================================

CODEDIR=/gpfs/labos/le2p/ctang/WRF_Data/ERA5
DATADIR=/gpfs/labos/le2p/ctang/WRF_Data/ERA5

# Set your python environment
cd $CODEDIR


function get_code
{

    DATE1=${1:-20670101}
    DATE2=${2:-20671231}

    echo $DATE1 $DATE2

    YY1=`echo $DATE1 | cut -c1-4`
    MM1=`echo $DATE1 | cut -c5-6`
    DD1=`echo $DATE1 | cut -c7-8`

    YY2=`echo $DATE2 | cut -c1-4`
    MM2=`echo $DATE2 | cut -c5-6`
    DD2=`echo $DATE2 | cut -c7-8`
    
    # for SWIO
    Nort=-2
    West=40
    Sout=-38
    East=74
    
    sed -e "s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;" GetERA5-sl.py > GetERA5-${DATE1}-${DATE2}-sl.py
    
    (python GetERA5-${DATE1}-${DATE2}-sl.py &)
    
    sed -e "s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;" GetERA5-pl.py > GetERA5-${DATE1}-${DATE2}-pl.py
    
    (python GetERA5-${DATE1}-${DATE2}-pl.py &)

    #mkdir -p ${DATADIR}/$YY1

    #mv ERA5-${DATE1}-${DATE2}-sl.grib ERA5-${DATE1}-${DATE2}-pl.grib ${DATADIR}/$YY1/

}

# ==================================
for year in 2021
do
    #for month in $(seq -w 1 12)
    for month in 05 07
    do

        first_day=${year}${month}01

        last_day=$(date --date="$year/$month/1 + 1 month - 1 day" "+%Y%m%d";)

        echo $first_day $last_day

        get_code $first_day $last_day

    done
done
# ==================================


exit 0
