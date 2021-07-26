import argparse

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--k", default=10, type=int, required=False,
                    help="choose top k entities")
parser.add_argument("--model", default='Full-Model', type=str, required=False,
                    help="choose the model to test")
parser.add_argument("--lr", default=0.0001, type=float, required=False,
                    help="initial learning rate")
parser.add_argument("--lrdecay", default=0.98, type=float, required=False,
                    help="learning rate decay every 5 epochs")
parser.add_argument("--test_by_type", default=False, required=False,
                    help="Evaluate the model by types (i.e., Sports, Socializing, Household, Personal Care, Eating)")
parser.add_argument("--val", default=False, required=False,
                    help="Evaluate the model")
parser.add_argument("--tes", default=False, required=False,
                    help="test the model")

args = parser.parse_args()
  