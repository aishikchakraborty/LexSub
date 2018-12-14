python main.py --sim-task --emb ../embeddings/emb_wikitext-2_Vanilla_300.pkl > results/sim_task_Vanilla.txt
echo 'Done with Vanilla Embeddings'
python main.py --sim-task --emb ../embeddings/emb_wikitext-2_WN_300.pkl > results/sim_task_WN.txt
echo 'Done with WN Embeddings'
