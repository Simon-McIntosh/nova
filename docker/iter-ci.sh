#!/bin/sh
wkdir=$1  # working directory
label=$2  # docker label
name="nova_$label"
. /usr/share/Modules/init/sh
module use /work/imas/etc/modules/all
# print hostname (debug)
hostname
# load udocker
module load udocker/1.1.4-intel-2020a-Python-2.7.18
docker_pytest="udocker --allow-root run --volume=$wkdir:/nova $name -m pytest /nova/tests/ --junitxml=/nova/tests/results.xml"
{ # try
# run from cached image
echo try from cached
$docker_pytest
} || { # except
# pull and build when cached image not found
echo listing local conatiners
udocker --allow-root ps
echo pulling twistersi/nova:$label
udocker --allow-root pull twistersi/nova:$label
echo creating image $name
udocker --allow-root create --name=$name twistersi/nova:$label
# run docker container (run pytest on nova/tests)
$docker_pytest
}
# print docker containers
udocker --allow-root ps

