#!/bin/bash
cd /Users/Sebi/Documents/grad_school/research/metabolism/gram/scripts/LinearSweep_181103_155821 

echo "Starting all batches at `date`"
while read P; do
echo "Processing batch ${P}"
python ./scripts/run_batch.py ${P} -N 1000 -S 0 -D 0
done < ./batches/index.txt 
echo "Completed all batches at `date`"
exit
