#!/bin/bash
cd /Users/Sebi/Documents/grad_school/research/metabolism/gram/scripts/LinearSweep_181103_155821 

while IFS=$'\t' read P
do
   JOB=`msub - << EOJ

#! /bin/bash
#MSUB -A p30653 
#MSUB -q short 
#MSUB -l walltime=10:00:00 
#MSUB -m abe 
#MSUB -o ./log/$(basename ${P})/out 
#MSUB -e ./log/$(basename ${P})/err 
#MSUB -N $(basename ${P}) 
#MSUB -l nodes=1:ppn=1 
#MSUB -l mem=1gb 

module load python/anaconda3.6
source activate ~/pythonenvs/metabolism_env

cd /Users/Sebi/Documents/grad_school/research/metabolism/gram/scripts/LinearSweep_181103_155821 

python ./scripts/run_batch.py ${P} -N 1000 -S 0 -D 0
EOJ
`

done < ./batches/index.txt 
echo "All batches submitted as of `date`"
exit
