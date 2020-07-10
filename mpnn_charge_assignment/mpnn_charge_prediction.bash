#!/bin/bash
# 
#  // ===============================
#  // AUTHOR      : Ali Raza (aliraza.ece@gmail.com)
#  // CREATE DATE : July 10, 2020
#  // PURPOSE     : A script to start a docker container and predict charges of a MOF
#  // SPECIAL NOTES: 
#  // ===============================
#  // Change History: 1.0: 
#  //
#  //==================================


# checkig if MOF name is provided
if [ "$1" == "" ]; then
    echo "error: provide mof name"
    exit 1
fi
echo ""
echo "making sure docker is installed"
# checking if docker is installed 
if [ -x "$(command -v docker)" ]; then
    : # docker is installed 
else
    echo "docker is not installed. visit https://docs.docker.com/get-docker/"
    exit 1
fi

echo "making sure docker image is present"
# making sure docker image for mpnn is present
if [[ "$(docker images -q razaa/mpnn_charge_prediction_image 2> /dev/null)" == "" ]]; then
    echo "docker image is not available. get the latest image by command 'razaa/mpnn_charge_prediction_image:version1' "
    exit 1
fi

# creating a container 
echo "creating a container"
docker run --name mpnn_charge_prediction_container -t -d razaa/mpnn_charge_prediction_image:version1 > /dev/null
# copying cif file to the container 
echo "copying $1.cif file to the container"
docker cp $1.cif mpnn_charge_prediction_container:/app/
# sending commands to the container to predict charges 
echo ""
docker exec mpnn_charge_prediction_container /julia/julia-1.4.2/bin/julia assign_mpnn_charges.jl "$1.cif"
# copying charge file to the current directory 
docker cp mpnn_charge_prediction_container:/app/$1_mpnn_charges.cif .
echo ""
echo "removing container"
echo "charges are stored in $1_mpnn_charges.cif"
docker rm  --force mpnn_charge_prediction_container > /dev/null
# end