
logfolder="$PWD/logs"
output_folder="$PWD/synthetic_output"
envvar="PYTHONPATH=$PWD:$PYTHONPATH"
script="$PWD/run_synthetic_experiments.py"
py="/home/nguyenm5/anaconda3/envs/pinot/bin/python"
 
for method in 'rr' 'rappor'
do
    for i in 2,100 5,50; do
    IFS=',' read k N <<< "${i}"

        for top in 'random' 'chain' 'star'
        do
            bsub -J spectral-rank -env "${envvar}"  -n 4 -R \
            "rusage[mem=4] span[hosts=1]" -W 96:00 -o "${logfolder}/${name}.stdout" -e "${logfolder}/${name}.stderr" \
            ${py} ${script} --top ${top} --k ${k} --method ${method} \
            --N ${N} --output_folder ${output_folder} 
        done
    done
done
