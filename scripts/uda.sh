#!/bin/bash
if [ $# -eq 0 ]
then
DEFSVN=trunk
else
DEFSVN=$1
fi
CURRF=$PWD
mkdir -p ${CURRF}/codacenv/include
mkdir -p ${CURRF}/codacenv/bin
mkdir -p ${CURRF}/codacenv/lib

export CODAC_ROOT=${CURRF}/codacenv

#svn co https://svnpub.iter.org/codac/iter/codac/dev/units/m-arch-common-base/${DEFSVN} marchcommonbase
#svn co https://svnpub.iter.org/codac/iter/codac/dev/units/m-arch-common-data/${DEFSVN} marchcommondata
#svn co https://svnpub.iter.org/codac/iter/codac/dev/units/m-arch-common-wrap/${DEFSVN} marchcommonwrap
#svn co https://svnpub.iter.org/codac/iter/codac/dev/units/m-uda-idam/${DEFSVN} mudaidam
#svn co https://svnpub.iter.org/codac/iter/codac/dev/units/m-uda-client/${DEFSVN} mudaclient

cd marchcommonbase
mkdir -p ${CURRF}/codacenv/include/dan
cp src/main/c/include/dan/common_defs_base.h ${CURRF}/codacenv/include/dan
cp src/main/c/include/dan/arch_limits.h ${CURRF}/codacenv/include/dan
cp src/main/c/include/dan/dan_build_config.h ${CURRF}/codacenv/include/dan
cp src/main/c/include/dan/dan_log_service.h ${CURRF}/codacenv/include/dan
cp src/main/c++/include/dan/common_defs_base.hpp ${CURRF}/codacenv/include/dan

cd src/main/c
make
cd ../c++
make
cd ../../..
cp ./target/lib/libdan_base* ${CURRF}/codacenv/lib

cd $CURRF
mkdir -p ${CURRF}/codacenv/include/dan/wrapper
cd marchcommondata
cp ./src/main/c++/include/dan/wrapper/dan_data_holder.h ../codacenv/include/dan/wrapper/
cd src/main/c
make
cd ../c++
make
cd ../../..
cp ./target/lib/libdan* ../codacenv/lib/
cd $CURRF

cd marchcommonwrap
cp ./src/main/c++/include/dan/wrapper/dan_data_srv_cpp.h ../codacenv/include/dan/wrapper/
cp ./src/main/c++/include/dan/wrapper/dan_correct_data_holders.h ../codacenv/include/dan/wrapper/
cp ./src/main/c++/include/dan/wrapper/numpy.i ../codacenv/include/dan/wrapper
cp ./src/main/c++/include/dan/wrapper/dan_data_srv_python.h ../codacenv/include/dan/wrapper
cp ./src/main/c++/include/dan/wrapper/numpy_srv.h ../codacenv/include/dan/wrapper
cd src/main/c++/wrapper-srv
make
cd ../../../..
cp ./target/lib/libdan_wrapper_srv* ${CURRF}/codacenv/lib
cd $CURRF

#cp CMakeLists.txt mudaidam/src/main/c/source/
#cp parseXML.cpp mudaidam/src/main/c/source/clientserver/
#cp parseXML.h mudaidam/src/main/c/source/clientserver/
cd mudaidam
cd src/main/c
cmake . -DCLIENT_ONLY=TRUE -DBUILD_SHARED_LIBS=ON -DSSLAUTHENTICATION=true -DNO_WRAPPERS=TRUE
make
#cp ./source/wrappers/java/UDA.jar ${CURRF}/codacenv/lib
cp --remove-destination ./source/client/libuda_client.* ${CURRF}/codacenv/lib
#cp --remove-destination ./source/wrappers/java/libuda_jni.* ${CURRF}/codacenv/lib
mkdir -p ${CURRF}/codacenv/include/uda
cp ./source/uda.h ${CURRF}/codacenv/include/uda/uda.h
mkdir -p ${CURRF}/codacenv/include/clientserver
cp ./source/clientserver/*.h ${CURRF}/codacenv/include/clientserver/
mkdir -p ${CURRF}/codacenv/include/client
cp ../../../../mudaidam/src/main/c/source/client/*.h ../../../../codacenv/include/client/
mkdir -p ${CURRF}/codacenv/include/structures
cp ./source/structures/*.h ${CURRF}/codacenv/include/structures/

cd $CURRF

cd mudaclient/src/main/c
make
cd ../c++
cd uda_client_reader_cpp
make
cd ../uda_client_reader_python3
mkdir -p uda_client_reader
make

#cd ../../python3
#make
cd ../../../../
cp ./target/lib/* ../codacenv/lib
mkdir -p ../codacenv/lib/uda_client_reader/
cp  src/main/c++/uda_client_reader_python3/uda_client_reader/*.so ../codacenv/lib/uda_client_reader/
cp  src/main/c++/uda_client_reader_python3/build/uda_client_reader_python.py ../codacenv/lib/uda_client_reader/
cp  src/main/c++/uda_client_reader_python3/UdaClientIterator.py ../codacenv/lib/uda_client_reader/
mkdir -p ../codacenv/python3/utils
cp -r src/main/python3/utils/* ../codacenv/python3/utils/
cd ../


