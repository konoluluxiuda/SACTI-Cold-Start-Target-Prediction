import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_mvcl_confirmation_data import export_bundle, check_partitions
from tools.mvcl_confirmation_inputs import load_confirmation_bundle
from tools.run_mvcl_cti_pilot import train
from util.mvcl_relation_scaled import RelationScaledMVCL
from util.mvcl_checkpoint import load_checkpoint


class LoaderTests(unittest.TestCase):
    def bundle(self, out):
        train, val, test = tuple([[c,'p',1],[c,'q',0]] for c in ('train','val','test'))
        ids = dict(herb=['h'], compound=['train','val','test'], protein=['p','q'], disease=['d'])
        export_bundle(out, ids, [('h',c) for c in ids['compound']], [('p','d')], train,val,test,
                      dict(check_partitions(train,val,test), dataset='tcmsuite', fold=2, split_seed=901026,
                           training_seed=52027, validation_seed=152027, source_hashes={}, sacti_partition_hashes_match=True))
        (out/'preparation_state.json').write_text('{"complete":true}')

    def test_nonfirst_fold_and_no_outer_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); self.bundle(out)
            data,pairs,labels=load_confirmation_bundle(out,'tcmsuite',2)
            self.assertEqual(pairs.tolist(), [[1,0],[1,1]])
            self.assertEqual(labels.tolist(), [1,0])
            self.assertNotIn('outer_test',data)
            with self.assertRaises(ValueError):
                load_confirmation_bundle(out,'tcmsuite',1)

    def test_incomplete_and_tampered_outer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); self.bundle(out)
            (out/'preparation_state.json').write_text('{"complete":false}')
            with self.assertRaises(ValueError):
                load_confirmation_bundle(out,'tcmsuite',2)
            (out/'preparation_state.json').write_text('{"complete":true}')
            records=json.loads((out/'outer_test.json').read_text());records[0][1]='q';records[1][1]='p'
            (out/'outer_test.json').write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                load_confirmation_bundle(out,'tcmsuite',2)

    def test_toy_training_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); data_dir=out/'data'; data_dir.mkdir(); self.bundle(data_dir)
            data,pairs,labels=load_confirmation_bundle(data_dir,'tcmsuite',2)
            kwargs=dict(dimension=4, seed=52027, block_size=2, sparse_graph=True)
            model=RelationScaledMVCL(data, **kwargs)
            training=out/'training';training.mkdir()
            config=dict(learning_rate=.001, epochs=1,eval_every=1,patience=5,min_delta=.0001,
                        contrast_weight=2.,reg_lambda=.5,scope='synthetic test only')
            result=train(model,pairs,labels,config,training)
            self.assertEqual(result['optimizer_steps'],1)
            self.assertFalse(result['outer_test_scored'])
            load_checkpoint(RelationScaledMVCL(data, **kwargs),training/'best_weights.pt')


if __name__ == '__main__':
    unittest.main()
