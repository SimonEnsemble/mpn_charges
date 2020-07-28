#!/bin/bash
module load slurm

methods=("iqeq" "ddec" "mpnn")

for method in ${methods[*]}
do
	for xtal in $(cat ./"$method"_left.txt)
	do
		echo "submitting job for $xtal"
		sbatch -J "$xtal"_"$method" -A simoncor -n 1 \
		-o ./"$method"_henry_outfiles/"$xtal.o" -e ./"$method"_henry_outfiles/"$xtal.e" \
		--time=96:00:00 \
		--export=xtal="$xtal",method="$method" run_henry.sh
	done
done
