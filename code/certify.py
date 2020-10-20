# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--confidence_measure", choices=["pred_score", "margin"], default="pred_score",
                    help="which confidence notion to use")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, args.confidence_measure)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\texp_cdf_00\texp_cdf_25\texp_cdf_50\texp_cdf_75\texp_cdf_100\texp_cdf_125\texp_cdf_150\t"
          "exp_00\texp_25\texp_50\texp_75\texp_100\texp_125\texp_150\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, exp_cdf_00, exp_cdf_25, exp_cdf_50, exp_cdf_75, exp_cdf_100, exp_cdf_125, exp_cdf_150, \
            exp_00, exp_25, exp_50, exp_75, exp_100, exp_125, exp_150 = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(
            i, label, prediction, exp_cdf_00, exp_cdf_25, exp_cdf_50, exp_cdf_75, exp_cdf_100, exp_cdf_125, exp_cdf_150,
            exp_00, exp_25, exp_50, exp_75, exp_100, exp_125, exp_150, correct, time_elapsed), file=f, flush=True)

    f.close()
