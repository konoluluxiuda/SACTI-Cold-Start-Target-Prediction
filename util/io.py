# Derived from https://github.com/tong87-bio/HDCTI (MIT License).
# Dataset file resolution and negative handling were extended here.
import os.path
from os import remove
from re import split
from .config import OptionConf
import random


NEGATIVE_FILE_CANDIDATES = ('ZERO_indices.txt', 'zero1.txt', 'zero.txt')


def resolve_optional_dataset_file(dataset_dir, candidates):
    for filename in candidates:
        path = os.path.join(dataset_dir, filename)
        if os.path.exists(path):
            return path
    return None


def sample_negative_records(positive_records, count, rng=None, deterministic=False):
    rng = rng or random
    left_ids = sorted({record[0] for record in positive_records})
    right_ids = sorted({record[1] for record in positive_records})
    positives = {(record[0], record[1]) for record in positive_records if float(record[2]) > 0}
    available = len(left_ids) * len(right_ids) - len(positives)
    if available < count:
        raise ValueError('Cannot generate %d negative samples from %d candidate pairs.' % (count, available))

    negatives = set()
    while len(negatives) < count:
        pair = (rng.choice(left_ids), rng.choice(right_ids))
        if pair in positives or pair in negatives:
            continue
        negatives.add(pair)
    negative_pairs = sorted(negatives) if deterministic else negatives
    return [[left, right, 0.0] for left, right in negative_pairs]


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def iterDataSet(conf, file, default_rating=1.0):
        ratingConfig = OptionConf(conf['ratings.setup'])
        order = ratingConfig['-columns'].strip().split()
        delim = ratingConfig['-delim'] if ratingConfig.contains('-delim') else ' |,|\t'
        with open(file) as f:
            if ratingConfig.contains('-header'):
                next(f, None)
            for lineNo, line in enumerate(f):
                values = split(delim, line.strip())
                if len(order) < 2:
                    raise ValueError('The rating file requires at least two columns: %s' % file)
                try:
                    left_id = values[int(order[0])]
                    right_id = values[int(order[1])]
                    rating = values[int(order[2])] if len(order) >= 3 else default_rating
                except (IndexError, ValueError) as exc:
                    raise ValueError('Invalid rating row %d in %s: %s' % (lineNo + 1, file, line.strip())) from exc
                yield [left_id, right_id, float(rating)]

    @staticmethod
    def readDataSet(conf, file, default_rating=1.0):
        return list(FileIO.iterDataSet(conf, file, default_rating=default_rating))


    @staticmethod
    def loadDataSet(conf, file, bTest=False):
        records = FileIO.readDataSet(conf, file, default_rating=1.0)
        trainingData = records if not bTest else []
        testData = records if bTest else []
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')

        dataset_dir = os.path.dirname(os.path.abspath(file))
        negative_count = len(testData if bTest else trainingData)
        zero_path = resolve_optional_dataset_file(dataset_dir, NEGATIVE_FILE_CANDIDATES)
        if zero_path:
            negative_pool = FileIO.readDataSet(conf, zero_path, default_rating=0.0)
            if negative_count > len(negative_pool):
                print('The negative file %s has only %d rows, but %d negatives are required.' %
                      (zero_path, len(negative_pool), negative_count))
                exit(-1)
            negative_records = random.sample(negative_pool, negative_count)
        else:
            print('No negative file found in %s. Generating %d negatives from positives.' %
                  (dataset_dir, negative_count))
            negative_records = sample_negative_records(testData if bTest else trainingData, negative_count)


        # with open('./dataset/TCMSP/zero1.txt') as f:
        #     ratings = f.readlines()
        #     # ignore the headline
        # if ratingConfig.contains('-header'):
        #      ratings = ratings[1:]
        # # order of the columns
        # order = ratingConfig['-columns'].strip().split()
        # delim = ' |,|\t'
        # if ratingConfig.contains('-delim'):
        #     delim=ratingConfig['-delim']
        # new_ratings=random.sample(ratings,56103)

        # with open('./dataset/Symmap/zero.txt') as f:
        #     ratings = f.readlines()
        # # ignore the headline
        # if ratingConfig.contains('-header'):
        #     ratings = ratings[1:]
        # # order of the columns
        # order = ratingConfig['-columns'].strip().split()
        # delim = ' |,|\t'
        # if ratingConfig.contains('-delim'):
        #     delim=ratingConfig['-delim']
        # new_ratings=random.sample(ratings,38043)

        for herbId, diseaseId, rating in negative_records:
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:
                trainingData.append([herbId, diseaseId, float(rating)])

        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadHerbList(filepath):
        herbList = []
        with open(filepath) as f:
            for line in f:
                herbList.append(line.strip().split()[0])
        return herbList
