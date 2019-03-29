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
                            output['test_ppl'] = '%.4f' % float(m.group(1))

                    if line in set(['men3k', 'simlex999', 'simverb3500', 'wordsim353_relatedness', 'hyperlex', 'hyperlex-nouns', 'hyperlex_test', \
                                    'syn_men3k', 'syn_simlex999', 'syn_simverb3500', 'syn_wordsim353_relatedness', 'syn_hyperlex', 'syn_hyperlex-nouns', 'syn_hyperlex_test', \
                                    'hyp_men3k', 'hyp_simlex999', 'hyp_simverb3500', 'hyp_wordsim353_relatedness', 'hyp_hyperlex', 'hyp_hyperlex-nouns', 'hyp_hyperlex_test', \
                                    'mer_men3k', 'mer_simlex999', 'mer_simverb3500', 'mer_wordsim353_relatedness', 'mer_hyperlex', 'mer_hyperlex-nouns', 'mer_hyperlex_test']):
                        wait_for_spearman = True
                        ws_bench = line

                    if ws_bench is not None and 'Spearman' in line:
                        wait_for_spearman = False
                        score = float(line.split(':')[1].strip())
                        output[ws_bench] = '%.4f' % (score)
                        ws_bench = None

            for ext, key in [('ner', 'test_f1-measure-overall'),
                                ('sst', 'test_accuracy'),
                                ('esim', 'test_accuracy'),
                                ('lex_relation_prediction', 'test_accuracy'),
                                ('bimpm', 'test_accuracy'),
                                ('decomposable', 'test_accuracy'),
                                ('bidaf', 'best_validation_em')]:

                ext_path = os.path.join(path, ext+'/metrics.json')
                if os.path.exists(ext_path):
                    with open(ext_path) as ext_stdout:
                        obj = json.load(ext_stdout)
                        output[ext] = '%.4f' % float(obj.get(key, '-1'))

            for name, prefix in [('hypernymysuite', ''), ('hyp_hypernymysuite', 'hyp_'), ('syn_hypernymysuite', 'syn_'), ('mer_hypernymysuite', 'mer_')]:
                ext_path = os.path.join(path, '%s.json' % name)
                if os.path.exists(ext_path):
                    with open(ext_path) as ext_stdout:
                        obj = json.load(ext_stdout)
                        output[prefix + 'dir_wbless'] = '%.4f' % obj['dir_wbless']['acc_test_inv']
                        output[prefix + 'dir_bibless'] = '%.4f' % obj['dir_bibless']['acc_test_inv']
                        output[prefix + 'dir_dbless'] = '%.4f' % obj['dir_dbless']['acc_test']
                        output[prefix + 'cor_hyperlex'] = '%.4f' % obj['cor_hyperlex']['rho_all']
                        output[prefix + 'siege_bless'] = '%.4f' % obj['siege_bless']['other']['ap_test']
                        output[prefix + 'siege_leds'] = '%.4f' % obj['siege_leds']['other']['ap_test']
                        output[prefix + 'siege_eval'] = '%.4f' % obj['siege_eval']['other']['ap_test']
                        output[prefix + 'siege_weeds'] = '%.4f' % obj['siege_weeds']['other']['ap_test']
                        output[prefix + 'siege_shwartz'] = '%.4f' % obj['siege_shwartz']['other']['ap_test']
            outputs.append(output)

        except Exception as e:
            print(e)
    json.dump(outputs, output_json, indent=2, separators=(',', ': '))
    fields = ['id', 'date', 'job_name', 'path', 'emb_file', 'test_ppl',\
                'men3k', 'wordsim353_relatedness', 'simlex999', 'simverb3500',  'hyperlex', 'hyperlex-nouns', 'hyperlex_test',  \
                'syn_men3k', 'syn_simlex999', 'syn_simverb3500', 'syn_wordsim353_relatedness', 'syn_hyperlex', 'syn_hyperlex-nouns', 'syn_hyperlex_test', \
                'hyp_men3k', 'hyp_simlex999', 'hyp_simverb3500', 'hyp_wordsim353_relatedness', 'hyp_hyperlex', 'hyp_hyperlex-nouns', 'hyp_hyperlex_test', \
                'mer_men3k', 'mer_simlex999', 'mer_simverb3500', 'mer_wordsim353_relatedness', 'mer_hyperlex', 'mer_hyperlex-nouns', 'mer_hyperlex_test', \
                'dir_wbless', 'dir_bibless', 'dir_dbless', 'cor_hyperlex', 'siege_bless', 'siege_leds', 'siege_eval', 'siege_weeds', 'siege_shwartz', \
                'syn_dir_wbless', 'syn_dir_bibless', 'syn_dir_dbless', 'syn_cor_hyperlex', 'syn_siege_bless', 'syn_siege_leds', 'syn_siege_eval', 'syn_siege_weeds', 'syn_siege_shwartz', \
                'hyp_dir_wbless', 'hyp_dir_bibless', 'hyp_dir_dbless', 'hyp_cor_hyperlex', 'hyp_siege_bless', 'hyp_siege_leds', 'hyp_siege_eval', 'hyp_siege_weeds', 'hyp_siege_shwartz', \
                'mer_dir_wbless', 'mer_dir_bibless', 'mer_dir_dbless', 'mer_cor_hyperlex', 'mer_siege_bless', 'mer_siege_leds', 'mer_siege_eval', 'mer_siege_weeds', 'mer_siege_shwartz', \
                'ner', 'sst', 'decomposable', 'bidaf', 'esim', 'bimpm', 'lex_relation_prediction']
    output_csv.write('%s\n' % ','.join(fields))
    for output in outputs:
        print(output)
        arr=[]
        for field in fields:
            arr.append(output.get(field, ''))
        output_csv.write('%s\n' % ','.join(arr))
