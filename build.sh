#!/usr/bin/env bash

set -e

[ -d OpenBLAS ] || git clone --recurse-submodules https://github.com/OpenMathLib/OpenBLAS --depth=1

cd OpenBLAS
if [ $(uname -s) = 'Darwin' ]; then
    # needs libomp gcc
    LDFLAGS="-L/opt/homebrew/opt/libomp/lib" CPPFLAGS="-I/opt/homebrew/opt/libomp/include" make -j$(sysctl -n hw.ncpu) TARGET=ARMV8 USE_OPENMP=1 NO_SVE=1 NUM_THREADS=32 FC=gfortran libs netlib shared
# TODO: linux else
fi
cp libopenblas*.a ../openblas.a
