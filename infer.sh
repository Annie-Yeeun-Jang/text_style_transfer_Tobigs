#!/bin/bash

echo "Convert Sentences' Style"
cd ./Style_Transformer
python3 infer.py
cd ..
cd ./Formal_Transformer
python3 infer.py
