#!/bin/bash

for FILE in ./Hammys*a1*gauss.h5; do python3 constant_fit.py --database $FILE; done &
for FILE in ./Hammys*a2*gauss.h5; do python3 constant_fit.py --database $FILE; done &
for FILE in ./Hammys*a3*gauss.h5; do python3 constant_fit.py --database $FILE; done &
for FILE in ./Hammys*a4*gauss.h5; do python3 constant_fit.py --database $FILE; done &
for FILE in ./Hammys*a5*gauss.h5; do python3 constant_fit.py --database $FILE; done &

echo "All 3 complete"
