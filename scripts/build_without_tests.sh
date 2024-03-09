#!/bin/bash

mkdir build
cd build
cmake -DUNIT_TESTS=OFF ..
make all
cd .. 
