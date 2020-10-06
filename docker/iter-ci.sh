#!/bin/sh
# initialize modules
. /usr/share/Modules/init/sh
module use /work/imas/etc/modules/all 
# load python and udocker
# module unload Python
module load Python/2.7.18-GCCcore-9.3.0
module load udocker/1.1.4-intel-2020a-Python-2.7.18
# run docker container
udocker --allow-root run --name nova twistersi/nova:base 

#-v **/:/nova twistersi/nova:base -m pytest --junitxml=**/tests/results.xml
# clearup
udocker rm nova

# -o junit_suite_name=pytest
#cat **/tests/results.xml

# docker run --name twistersi/nova:base -v C:/Users/mcintos/Work/Code/nova:/nova nova -m pytest /nova/tests/

