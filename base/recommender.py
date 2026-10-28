# Derived from https://github.com/tong87-bio/HDCTI (MIT License).
# Evaluation metrics and strict checkpoint handling were extended here.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rating import Rating
from util.io import FileIO
from util.config import OptionConf
from util.log import Log
from os.path import abspath
import os
from time import strftime,localtime,time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import random

from sklearn.metrics import mean_squared_error, matthews_corrcoef
class Recommender(object):
    def __init__(self, conf, trainingSet, testSet, fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = Rating(self.config, trainingSet, testSet)
        self.validationData = []
        self.validationDataByState = {}
        self.validationAggregation = None
        # print(self.data.herb)
        # print(self.data.disease)
        # print(len(self.data.disease))
        self.foldInfo = fold
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.num_herbs, self.num_diseases, self.train_size = self.data.trainingSize()
        self.num_compounds, self.num_proteins = self.data.cpSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.modelName, self.modelName + self.foldInfo + ' ' + currentTime)
        # save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k + '=' + self.config[k])

    def readConfiguration(self):
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()

    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:', self.config['model.name'])
        if self.config.contains('model.variant'):
            print('Model variant:', self.config['model.variant'])
        print('Ratings dataset:', abspath(self.config['datapath']))
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        train_compounds = len({str(record[0]) for record in self.data.trainingData})
        train_proteins = len({str(record[1]) for record in self.data.trainingData})
        test_compounds = len({str(record[0]) for record in self.data.testData})
        test_proteins = len({str(record[1]) for record in self.data.testData})
        print('Entity universe: herbs %d, compounds %d, proteins %d, diseases %d' % (
            len(self.data.herb), len(self.data.compound),
            len(self.data.protein), len(self.data.disease)
        ))
        print('Training pairs: %d (active compounds %d, proteins %d)' % (
            len(self.data.trainingData), train_compounds, train_proteins
        ))
        print('Test pairs: %d (active compounds %d, proteins %d)' % (
            len(self.data.testData), test_compounds, test_proteins
        ))
        print('=' * 80)
        # print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr += key[1:] + ':' + args[key] + '  '
            print('Specific parameters:', parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def trainModel(self):
        'build the model (for model-based Models )'
        pass

    def trainModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    # for rating prediction
    def predictForRating(self, u, i):
        pass

    # for disease prediction
    def predictForRanking(self, u):
        pass

    def predictForPairs(self, compound_indices, protein_indices):
        candidates = self.predictForRanking()
        return candidates[
            np.asarray(compound_indices, dtype=np.int64),
            np.asarray(protein_indices, dtype=np.int64),
        ]

    def checkRatingBoundary(self, prediction):
        pass

    def evalRatings(self):
        pass

    def softmax(self, x):
        #    orig_shape=x.shape
        if len(x.shape) > 1:
            tmp = np.max(x, axis=1)
            x -= tmp.reshape((x.shape[0], 1))
            x = np.exp(x)
            tmp = np.sum(x, axis=1)
            x /= tmp.reshape((x.shape[0], 1))
        else:
            tmp = np.max(x)
            x -= tmp
            x = np.exp(x)
            tmp = np.sum(x)
            x /= tmp
        return x



    def evalRanking(self):
        print('recommender evalRanking-------------------------------------------------------')
        compound_indices = []
        protein_indices = []
        lable_mat = []

        for herb in self.data.testSet_h:
            diseaselist = self.data.testSet_h[herb].keys()
            for disease in diseaselist:
                compound_id = self.data.compound[herb]
                protein_id = self.data.protein[disease]
                compound_indices.append(compound_id)
                protein_indices.append(protein_id)
                lable_mat.append(self.data.testSet_h[herb][disease])

        print('hdctipredict----------------------------------------------------------------------------')
        herb_mat = np.asarray(
            self.predictForPairs(compound_indices, protein_indices), dtype=float
        )
        herb_mat = 1 / (1 + np.exp(-np.clip(herb_mat, -50, 50)))

        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        data0 = {
            'label': lable_mat,
            'predict': herb_mat,

        }
        dataframe0 = pd.DataFrame(data0)
        os.makedirs('./results/cv', exist_ok=True)
        variant = self.config['model.variant'] if self.config.contains('model.variant') else self.modelName
        dataframe0.to_csv('./results/cv/' + variant + '@' + currentTime + self.foldInfo + '.txt',
                          columns=['label', 'predict'], index=False, header=None)

        labels = np.asarray(lable_mat, dtype=int)
        scores = herb_mat
        if not np.all(np.isfinite(scores)):
            raise ValueError('Prediction scores contain NaN or infinity. The model training is numerically unstable.')
        predictions = (scores >= 0.5).astype(int)

        metrics = [
            ('AUC', roc_auc_score(labels, scores)),
            ('AUPR', average_precision_score(labels, scores)),
            ('Recall', recall_score(labels, predictions, zero_division=0)),
            ('Precision', precision_score(labels, predictions, zero_division=0)),
            ('F1-score', f1_score(labels, predictions, zero_division=0)),
        ]
        self.measure = []
        for name, value in metrics:
            print('%s: %s' % (name, value))
            self.measure.append('%s:%s' % (name, value))


    def execute(self):
        self.readConfiguration()
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        # load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' % self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' % self.foldInfo)
            self.initModel()
            print('Building Model %s...' % self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.trainModel_tf()
                else:
                    self.trainModel()
            except ImportError:
                self.trainModel()
        outer_test_enabled = True
        if self.config.contains('evaluation.outer.test'):
            outer_test_value = str(self.config['evaluation.outer.test']).strip().lower()
            if outer_test_value in ('0', 'false', 'no', 'off'):
                outer_test_enabled = False
            elif outer_test_value not in ('1', 'true', 'yes', 'on'):
                raise ValueError(
                    'Invalid boolean value for evaluation.outer.test: %s' %
                    self.config['evaluation.outer.test']
                )
        if not outer_test_enabled:
            summary = getattr(self, 'early_stopping_summary', None)
            if not summary:
                raise ValueError(
                    'evaluation.outer.test=False requires early stopping with a validation metric.'
                )
            aggregation = summary.get('aggregation')
            metric_name = 'Validation-' + summary['metric'].upper()
            if aggregation == 'macro_support_states':
                metric_name = 'Validation-Macro-' + summary['metric'].upper()
            print(
                'Outer test evaluation skipped for model selection; best %s: %.6f at epoch %d.' % (
                    metric_name.replace('Validation-', ''),
                    summary['best_value'],
                    summary['best_epoch'],
                )
            )
            self.measure = ['%s:%s' % (metric_name, summary['best_value'])]
            return self.measure
        print('Predicting %s...' % self.foldInfo)
        self.evalRanking()
        # self.calcAccuracy()
        if self.isSaveModel:
            print('Saving model %s...' % self.foldInfo)
            self.saveModel()
        return self.measure
