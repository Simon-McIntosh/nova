#!/bin/bash

git clone https://github.com/ukaea/UDA.git -b release/2.7.4 uda
cd uda

cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Debug -DSSLAUTHENTICATION=ON -DCLIENT_ONLY=OFF -DENABLE_CAPNP=ON -DCMAKE_INSTALL_PREFIX=install

cmake --build build/ -j 2
cmake --install build/

cd install/etc
touch machine.d/udatrain.cfg
sed -i '5i export UDAHOSTNAME=udatrain' uda/install/etc/udaserver.cfg

git clone ssh://git@git.iter.org/scm/imas/uda-plugins.git -b release/1.4.0
cd uda-plugins

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HOME/Code/uda/install/lib/pkgconfig
cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug -DBUILD_PLUGINS=imas -DCMAKE_INSTALL_PREFIX=$HOME/Code/uda/install

cmake --build build/ -j 2
cmake --install build/
