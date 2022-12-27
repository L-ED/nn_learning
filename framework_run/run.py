import sys
sys.path.insert(0, "/LyginE/projects/paradigma/")
from quant_framework_new import Experimentator
import argparse

parser = argparse.ArgumentParser(description="Experiment run")
parser.add_argument("-c", "--config")

args = parser.parse_args()

exps = Experimentator(args.config)
exps()