import glob
import json
import os
import re
import sys

after=int(sys.argv[1]) if len(sys.argv) > 1 else 0
with open('lm_wn_machine_assignments.txt', 'r') as paths, \
    open('output_results.txt', 'w'):
    for line in paths:
        try:
            sub_id, machine, path = line.strip().split(' - ')

            sub_id = int(sub_id)

            if sub_id < after:
                continue

            # print(path)
            emb_file_paths = glob.glob('/'.join([path, 'emb_*.pkl']))
            if len(emb_file_paths) == 0:
                continue
            for emb_file_path in emb_file_paths:
                output = []
                output.append(sub_id)
                output.append(emb_file_path)

                ws_bench = None
                with open(os.path.join(path, 'std.out')) as std_out:
                    for line in std_out:
                        line = line.strip()
                        if "test ppl" in line:
                            m = re.search('test ppl[ ]+([0-9.]+)', line)
                            if m is not None:
                                output.append('test ppl: %.4f' % float(m.group(1)))

                        if line in set(['men3k', 'simlex999', 'simverb3500', 'wordsim353_relatedness']):
                            wait_for_spearman = True
                            ws_bench = line

                        if ws_bench is not None and 'Spearman' in line:
                            wait_for_spearman = False
                            score = float(line.split(':')[1].strip())
                            output.append('%s: %.4f' % (ws_bench, score))
                            ws_bench = None

                for ext, key in [('ner', 'test_f1-measure-overall'), ('sst', 'test_accuracy'), ('esim', 'test_accuracy'),('bidaf', 'best_validation_em')]:
                    ext_path = os.path.join(path, ext+'/metrics.json')
                    if os.path.exists(ext_path):
                        with open(ext_path) as ext_stdout:
                            obj = json.load(ext_stdout)
                            output.append('%s: %.4f' % (ext, float(obj[key])))
                if len(output) > 2:
                    print(output)
        except Exception as e:
            print(e)
