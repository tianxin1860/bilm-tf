#!/bin/bash
echo "==============JOB BEGIN============"

# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun sh run.sh
mpirun -bind-to none ./setup.sh

echo "===============SETUP DONE=========="

mpirun -bind-to none ./run.sh
echo "===============JOB END============="
