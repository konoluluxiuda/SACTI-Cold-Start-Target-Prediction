import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_mvcl_confirmation_loader import LoaderTests
from tools import evaluate_mvcl_confirmation as evaluator
from tools.run_mvcl_confirmation import jobs_for, summarize
from tools.run_mvcl_cti_pilot import train
from tools.mvcl_confirmation_inputs import load_confirmation_bundle
from util.mvcl_relation_scaled import RelationScaledMVCL


class ConfirmationTests(unittest.TestCase):
    def test_all_training_before_any_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=jobs_for(dict(datasets=['tcmsuite','tcmsp','symmap','etcm_mention10'],folds=[1,2,3,4,5]),Path(tmp),'python')
            self.assertEqual(sum(j['stage']=='train' for j in jobs),20)
            self.assertEqual(sum(j['stage']=='evaluate' for j in jobs),20)
            self.assertTrue(all(j['stage']=='evaluate' for j in jobs[-20:]))
            self.assertFalse(any(j['stage']=='evaluate' for j in jobs[:-20]))
            summarize(Path(tmp),jobs)
            report=json.loads((Path(tmp)/'summary.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['completed_outer_folds'],0)

    def test_toy_fixed_checkpoint_outer_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data_dir=root/'data';data_dir.mkdir()
            LoaderTests().bundle(data_dir)
            data,vp,vy=load_confirmation_bundle(data_dir,'tcmsuite',2)
            kwargs=dict(dimension=4,seed=52027,block_size=2,sparse_graph=True)
            model=RelationScaledMVCL(data,**kwargs)
            training=root/'training';training.mkdir()
            config=dict(model=kwargs,learning_rate=.001,epochs=1,eval_every=1,patience=5,
                        min_delta=.0001,contrast_weight=2.,reg_lambda=.5,scope='synthetic only')
            selected=train(model,vp,vy,config,training)
            formal=root/'results/mvcl_formal_confirmation_protocol_v1/seal.json'
            formal.parent.mkdir(parents=True)
            formal.write_text(json.dumps(dict(source_hashes={},protocol=dict(ranking_ks=[10,20]))))
            for name in ('tools/evaluate_mvcl_ranking.py','util/checkpoint_ranking.py',
                         'tools/evaluate_non_neural_cold_start_baselines.py'):
                f=root/name;f.parent.mkdir(exist_ok=True);f.write_text('test provenance fixture')
            seal=dict(protocol='mvcl_scaled_confirmation_fold_v1',dataset='tcmsuite',fold=2,
                dry_run=False,outer_test_scored=False,confirmation_seal_sha256=evaluator.digest(formal),
                bundle_source_hashes={},code_hashes={},data=str(data_dir),config=config,
                data_fingerprint=model.checkpoint_fingerprint,
                **{k:data['report'][k] for k in ('train_sha256','validation_sha256','outer_test_sha256')})
            state=dict(training_complete=True,dataset='tcmsuite',fold=2,best_epoch=selected['best_epoch'],outer_test_scored=False)
            (training/'seal.json').write_text(json.dumps(seal))
            (training/'confirmation_state.json').write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                evaluator.check_completion(seal,selected,dict(state,training_complete=False),training)
            with self.assertRaises(ValueError):
                evaluator.check_completion(seal,dict(selected,checkpoint=str(training/'other.pt')),state,training)
            before=evaluator.digest(training/'best_weights.pt')
            cwd=Path.cwd()
            try:
                os.chdir(root)
                with patch.object(evaluator,'ROOT',root):
                    evaluator.run(argparse.Namespace(run=training,output=root/'evaluation',device='cpu'))
            finally:
                os.chdir(cwd)
            result=json.loads((root/'evaluation/summary.json').read_text())
            self.assertEqual(result['protocol']['training_optimizer_steps'],0)
            self.assertTrue(result['protocol']['outer_test_scored'])
            self.assertEqual(set(result['metrics']),{'mvcl','prototype','jaccard','raw_equal'})
            self.assertIn('Precision@10',result['metrics']['mvcl'])
            self.assertEqual(before,evaluator.digest(training/'best_weights.pt'))


if __name__=='__main__':
    unittest.main()
