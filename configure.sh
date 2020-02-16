#!/bin/bash

mkdir lib
cp -r ~/projects/grl/addons/python/share/grlenv/ lib/
cp -r ~/projects/grl/build/libgrl.so lib/
cp --remove-destination ~/projects/grl/build/libyaml-cpp.so.0.5 lib/
cp -r ~/projects/grl/build/grlpy.cpython-36m-x86_64-linux-gnu.so lib/
#cp -r ~/projects/grl/build/libyaml-cpp.so.0.5.3 lib/


