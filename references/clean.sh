#!/bin/bash

rm papers.pub
rm papers.bib

for file in *.bib
do
    echo $file
    publish import $file

done

publish export papers.bib
