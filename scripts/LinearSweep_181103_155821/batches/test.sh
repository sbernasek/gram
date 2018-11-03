#!/bin/bash
cd /Users/Sebi/Documents/grad_school/research/metabolism/gram/scripts/LinearSweep_181103_155821
echo "Starting."
while IFS=$'\t' read P
do


#batch_id=$(basename ${P})
batch_id=$(echo $(basename ${P}) | cut -f 1 -d '.')

#batch_id=$(echo ${P} | cut -f 1 -d '.')
echo ${batch_id}
#echo "./log/$(basename ${P})/out"
done < ./batches/index.txt
echo "Done."
exit
