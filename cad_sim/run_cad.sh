#!/bin/bash


echo "Running freecad in Docker "

#if want to debugg in bash mode
#docker run  -it --rm --name tri -v ${PWD}:/home/ubuntu/butterfly adi:sone

docker run  -it --rm --name tri -v ${PWD}/cad_sim:/home/ubuntu/butterfly adi:sone 

echo "*** ALL DONE -- ALL DONE ***"
