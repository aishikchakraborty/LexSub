for syn_ratio in 0.001 0.01 0.1 1
do
    for hyp_ratio in 0.001 0.01 0.1 1
    do
        for mer_ratio in 0.001 0.01 0.1 1
        do
            syn=True hyp=True mer=true data=glove mdl=retro syn_ratio=${syn_ratio} hyp_ratio=${hyp_ratio} mer_ratio=${mer_ratio} lr=0.5 scripts/run_once.sh
        done
    done
done
