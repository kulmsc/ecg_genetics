#! /bin/bash -l
 
#SBATCH --partition=panda-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=nn-tuning
#SBATCH --time=48:00:00   # HH/MM/SS
#SBATCH --mem=76G
#SBATCH --gres=gpu


source ~/.bashrc
spack load -r /cypfv4n miniconda3@4.3.14%gcc@6.3.0 arch=linux-centos7-x86_64

#echo "Starting at:" `date` >> hello_slurm_output.txt
source activate /home/kulmsc/.conda/envs/ecg

echo $SLURM_JOB_NODELIST

rm genetic_data.pl
#python3 temp_scott_main.py
rm pickup_params.pl
for i in {1..20};do
        echo "GENERATION $i"
	python ga_main.py
        echo "done generation"
done


exit
