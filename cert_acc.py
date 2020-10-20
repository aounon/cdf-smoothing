import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cert_file", type=str, help="File containing certified radii")
parser.add_argument("exp_threshold", type=float, help="Threshold for expected top class score")
args = parser.parse_args()

if __name__ == "__main__":
    exp_cdf_00 = []
    exp_cdf_25 = []
    exp_cdf_50 = []
    exp_cdf_75 = []
    exp_cdf_100 = []
    exp_cdf_125 = []
    exp_cdf_150 = []
    exp_00 = []
    exp_25 = []
    exp_50 = []
    exp_75 = []
    exp_100 = []
    exp_125 = []
    exp_150 = []
    correct = []

    with open(args.cert_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter='\t')
        is_first_row = True
        for row in csvReader:
            if is_first_row:
                is_first_row = False
            else:
                exp_cdf_00.append(float(row[3]))
                exp_cdf_25.append(float(row[4]))
                exp_cdf_50.append(float(row[5]))
                exp_cdf_75.append(float(row[6]))
                exp_cdf_100.append(float(row[7]))
                exp_cdf_125.append(float(row[8]))
                exp_cdf_150.append(float(row[9]))
                exp_00.append(float(row[10]))
                exp_25.append(float(row[11]))
                exp_50.append(float(row[12]))
                exp_75.append(float(row[13]))
                exp_100.append(float(row[14]))
                exp_125.append(float(row[15]))
                exp_150.append(float(row[16]))
                correct.append(int(row[17]))

    exp_cdf_00 = np.array(exp_cdf_00)
    exp_cdf_25 = np.array(exp_cdf_25)
    exp_cdf_50 = np.array(exp_cdf_50)
    exp_cdf_75 = np.array(exp_cdf_75)
    exp_cdf_100 = np.array(exp_cdf_100)
    exp_cdf_125 = np.array(exp_cdf_125)
    exp_cdf_150 = np.array(exp_cdf_150)
    exp_00 = np.array(exp_00)
    exp_25 = np.array(exp_25)
    exp_50 = np.array(exp_50)
    exp_75 = np.array(exp_75)
    exp_100 = np.array(exp_100)
    exp_125 = np.array(exp_125)
    exp_150 = np.array(exp_150)
    correct = np.array(correct)

    num_ex = correct.size
    cert_exp_cdf_00 = float(np.sum(correct[exp_cdf_00 > args.exp_threshold]))/num_ex
    cert_exp_cdf_25 = float(np.sum(correct[exp_cdf_25 > args.exp_threshold]))/num_ex
    cert_exp_cdf_50 = float(np.sum(correct[exp_cdf_50 > args.exp_threshold]))/num_ex
    cert_exp_cdf_75 = float(np.sum(correct[exp_cdf_75 > args.exp_threshold]))/num_ex
    cert_exp_cdf_100 = float(np.sum(correct[exp_cdf_100 > args.exp_threshold]))/num_ex
    cert_exp_cdf_125 = float(np.sum(correct[exp_cdf_125 > args.exp_threshold]))/num_ex
    cert_exp_cdf_150 = float(np.sum(correct[exp_cdf_150 > args.exp_threshold]))/num_ex
    cert_exp_00 = float(np.sum(correct[exp_00 > args.exp_threshold]))/num_ex
    cert_exp_25 = float(np.sum(correct[exp_25 > args.exp_threshold]))/num_ex
    cert_exp_50 = float(np.sum(correct[exp_50 > args.exp_threshold]))/num_ex
    cert_exp_75 = float(np.sum(correct[exp_75 > args.exp_threshold]))/num_ex
    cert_exp_100 = float(np.sum(correct[exp_100 > args.exp_threshold]))/num_ex
    cert_exp_125 = float(np.sum(correct[exp_125 > args.exp_threshold]))/num_ex
    cert_exp_150 = float(np.sum(correct[exp_150 > args.exp_threshold]))/num_ex

    print("Expected score threshold = %f" % args.exp_threshold)
    print("\n")
    print("Certified accuracy (using cdf bound) at radius 0.00 = %f" % cert_exp_cdf_00)
    print("Certified accuracy (using cdf bound) at radius 0.25 = %f" % cert_exp_cdf_25)
    print("Certified accuracy (using cdf bound) at radius 0.50 = %f" % cert_exp_cdf_50)
    print("Certified accuracy (using cdf bound) at radius 0.75 = %f" % cert_exp_cdf_75)
    print("Certified accuracy (using cdf bound) at radius 1.00 = %f" % cert_exp_cdf_100)
    print("Certified accuracy (using cdf bound) at radius 1.25 = %f" % cert_exp_cdf_125)
    print("Certified accuracy (using cdf bound) at radius 1.50 = %f" % cert_exp_cdf_150)
    print("\n")
    print("Certified accuracy (without cdf bound) at radius 0.00 = %f" % cert_exp_00)
    print("Certified accuracy (without cdf bound) at radius 0.25 = %f" % cert_exp_25)
    print("Certified accuracy (without cdf bound) at radius 0.50 = %f" % cert_exp_50)
    print("Certified accuracy (without cdf bound) at radius 0.75 = %f" % cert_exp_75)
    print("Certified accuracy (without cdf bound) at radius 1.00 = %f" % cert_exp_100)
    print("Certified accuracy (without cdf bound) at radius 1.25 = %f" % cert_exp_125)
    print("Certified accuracy (without cdf bound) at radius 1.50 = %f" % cert_exp_150)
