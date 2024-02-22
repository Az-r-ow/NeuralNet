#!/bin/bash

mkdir build
cd build
cmake -DPYBIND_BUILD=OFF ..
make all
cd .. 
