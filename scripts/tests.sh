#!/bin/bash

# Accessing the tests folder
cd build/tests

# looping of the tests 
for file in ./*; do 
  if [[ -f "$file" && -x "$file" ]]; then
    echo "Running : $file"
    # executing the test files
    $file
  fi 
done 

# Navigating back to the main folder
cd ../..
