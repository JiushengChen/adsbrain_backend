#!/bin/bash

rm -rf build
mkdir build
cd build
cmake \
  -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
  -DTRITON_COMMON_REPO_TAG:STRING=r22.05 \
  -DTRITON_CORE_REPO_TAG:STRING=r22.05 \
  -DTRITON_BACKEND_REPO_TAG:STRING=r22.05 \
  -DCMAKE_BUILD_TYPE:STRING=Release \
  ..

make -j install
cd ../

