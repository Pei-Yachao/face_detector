#!/bin/bash
for((i=0; i<$(ls ../config | wc -l); i++)); do
    echo -e "../config/" >> file.txt
done
