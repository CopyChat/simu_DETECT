#!/bin/bash - 
#======================================================
#
#          FILE: get_output.sh
# 
USAGE="./get_output.sh"
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
#  ORGANIZATION: 
#       CREATED: 09/01/2021 17:35
#      REVISION: 1.0
#=====================================================
set -o nounset           # Treat unset variables as an error
. ~/Shell/functions.sh   # ctang's functions

while getopts ":tf:" opt; do
    case $opt in
        t) TEST=1 ;;
        f) file=$OPTARG;;
        \?) echo $USAGE && exit 1
    esac
done
shift $(($OPTIND - 1))
#=================================================== 

for mon in 06 07 08
do
    echo $mon
    rsync -aruxHPS ctb19335@ssh-ccub.u-bourgogne.fr:/work/crct/ctb19335/Modeling/DETECT/wrf/wrfout*d03_2021-${mon}* \
    /Users/ctang/Microsoft_OneDrive/OneDrive/CODE/simu_DETECT/wrf/output_for_detect/

done

