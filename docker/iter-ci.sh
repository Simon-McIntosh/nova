#!/bin/sh
wkdir=$1  # working directory
. /usr/share/Modules/init/sh
module use /work/imas/etc/modules/all
# print hostname (debug)
hostname
# load udocker
module load udocker/1.1.4-intel-2020a-Python-2.7.18
# pull and build (should use cached images here)
udocker --allow-root pull twistersi/nova:base
udocker --allow-root create --name=nova_base twistersi/nova:base
# run docker container (run pytest on nova/tests)
udocker --allow-root run --volume=$wkdir:/nova nova_base -m pytest /nova/tests/ --junitxml=/nova/tests/results.xml
# print docker containers
udocker --allow-root ps

