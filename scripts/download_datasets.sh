#!/bin/bash
mkdir -p data/
cd data/
wget https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz  
wget https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz
tar xzf Pubmed-Diabetes.tgz
tar xzf citeseer.tgz
