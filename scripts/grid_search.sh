for n_margin in 0.1 0.25 0.5 1
do
    for syn_ratio in 0.001 0.01 0.1
    do
        for hyp_ratio in 0.001 0.01 0.1
        do
            for mer_ratio in 0.001 0.01 0.1
            do
                output_dir_prefix=hyperparam_sweep_2 syn=True hyp=True mer=True num_ext_runs=1 data=glove mdl=retro syn_ratio=${syn_ratio} hyp_ratio=${hyp_ratio} mer_ratio=${mer_ratio} n_margin=${n_margin} lr=0.5 scripts/run_once.sh
            done
        done
    done
done
