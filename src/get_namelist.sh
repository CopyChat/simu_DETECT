#!/bin/bash - 
#======================================================
#
#          FILE: get_domain_ccub.sh
# 
USAGE="./get_namelist.sh"
# 
#   DESCRIPTION: to get namelist files from CCuB
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
#  ORGANIZATION: 
#       CREATED: 08/30/2021 16:36
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
#=================================================== p
rsync -aruxHPS ctb19335@ssh-ccub.u-bourgogne.fr:/work/crct/ctb19335/Modeling/DETECT/wps/namelist.wps \
    /Users/ctang/Microsoft_OneDrive/OneDrive/CODE/simu_DETECT/wrf/namelist/

rsync -aruxHPS ctb19335@ssh-ccub.u-bourgogne.fr:/work/crct/ctb19335/Modeling/DETECT/wrf/namelist.input \
    /Users/ctang/Microsoft_OneDrive/OneDrive/CODE/simu_DETECT/wrf/namelist/

rsync -aruxHPS ctb19335@ssh-ccub.u-bourgogne.fr:/work/crct/ctb19335/Modeling/DETECT/wrf/output_d0?.txt \
    /Users/ctang/Microsoft_OneDrive/OneDrive/CODE/simu_DETECT/wrf/output_setup/
