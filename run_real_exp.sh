
logfolder="$PWD/logs"
output_folder="$PWD/real_output"
envvar="PYTHONPATH=$PWD:$PYTHONPATH"
script="$PWD/run_real_experiment.py"
py="/home/nguyenm5/anaconda3/envs/pinot/bin/python"
 
for method in 'rr' 'rappor'
do

        for data in 'youtube'
        do
            name="${data}_${method}"
            bsub -J real-asr -env "${envvar}"  -n 4 -R \
            "rusage[mem=4] span[hosts=1]" -W 96:00 -o "${logfolder}/${name}.stdout" -e "${logfolder}/${name}.stderr" \
            ${py} ${script} --method ${method} \
            --data ${data} \
            --output_folder ${output_folder} 
        done
done
