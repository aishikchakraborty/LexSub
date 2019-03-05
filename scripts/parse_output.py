import glob
import json
import os
import re
import sys
import json
version=sys.argv[1] if len(sys.argv) > 1 else 'v2'
print(version)
with open('lm_wn_machine_assignments%s.txt' % ('-v2' if version=='v2' else ''), 'r') as paths, \
    open('output/output_results.csv', 'w') as output_csv, \
    open('output/output_results.json', 'w') as output_json:
    outputs=[]
    for line in paths:
        try:
            if version == 'v2':
                date, job_name, sub_id, machine, path = line.strip().split(' - ')
                output = {
                            'id': sub_id,
                            'date': date,
                            'job_name': job_name,
                            'path': path
                         }
            else:
                sub_id, machine, path = line.strip().split(' - ')
                output = {
                            'id': sub_id,
                            'date': None,
                            'job_name': None,
                            'path': path
                         }

            sub_id = int(sub_id)

            emb_file_paths = glob.glob('/'.join([path, 'emb_*.pkl']))
            for emb_file_path in emb_file_paths:
                if re.search('emb_(glove|wiki).*.pkl', emb_file_path):
                    output['emb_file'] = emb_file_path.split('/')[-1].split('.')[0]

            ws_bench = None
            with open(os.path.join(path, 'std.out')) as std_out:
                for line in std_out:
                    line = line.strip()
                    if "test ppl" in line:
                        m = re.search('test ppl[ ]+([0-9.]+)', line)
                        if m is not None:
                            output['test ppl'] = '%.4f' % float(m.group(1))

                    if line in set(['men3k', 'simlex999', 'simverb3500', 'wordsim353_relatedness', 'hyperlex', 'hyperlex-nouns', 'hyperlex_test']):
                        wait_for_spearman = True
                        ws_bench = line

                    if ws_bench is not None and 'Spearman' in line:
                        wait_for_spearman = False
                        score = float(line.split(':')[1].strip())
                        output[ws_bench] = '%.4f' % (score)
                        ws_bench = None

            for ext, key in [('ner', 'test_f1-measure-overall'), ('sst', 'test_accuracy'), ('esim', 'test_accuracy'),('bidaf', 'best_validation_em')]:
                ext_path = os.path.join(path, ext+'/metrics.json')
                if os.path.exists(ext_path):
                    with open(ext_path) as ext_stdout:
                        obj = json.load(ext_stdout)
                        output[ext] = '%.4f' % float(obj[key])

            outputs.append(output)

        except Exception as e:
            print(e)
    json.dump(outputs, output_json, sort_keys=True,indent=2, separators=(',', ': '))
    fields = ['id', 'date', 'job_name', 'path', 'emb_file', 'men3k', 'wordsim353_relatedness', 'simlex999', 'simverb3500',  'hyperlex', 'hyperlex-nouns', 'hyperlex_test',  'ner', 'sst', 'esim', 'bidaf']
    output_csv.write('%s\n' % ','.join(fields))
    for output in outputs:
        print(output)
        arr=[]
        for field in fields:
            arr.append(output.get(field, ''))
        output_csv.write('%s\n' % ','.join(arr))
