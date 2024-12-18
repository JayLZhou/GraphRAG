from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
args = parser.parse_args()

opt = Config.parse(Path(args.opt), dataset_name = args.dataset_name)
digimon = GraphRAG(config=opt)


def check_dirs(opt):
    # For each query, save the results in a separate directory
    result_dir = os.path.join(
        opt.working_dir, opt.exp_name, "Results"
    )
    # Save the current used config in a separate directory
    config_dir = os.path.join(
        opt.working_dir, opt.exp_name, "Configs"
    )
    # Save the metrics of entire experiment in a separate directory
    metric_dir = os.path.join(
        opt.working_dir, opt.exp_name, "Metrics"
    )
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_name = args.opt[args.opt.rindex("/") + 1:]
    basic_name = os.path.join(args.opt.split("/")[0], "Config2.yaml")
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))


check_dirs(opt)
# 
query_dataset = RAGQueryDataset(data_dir=os.path.join(opt.data_root, opt.dataset_name))
corpus = query_dataset.get_corpus()
# print(f"length of corpus: {len(corpus)}")
corpus = corpus[:100]
# asyncio.run(digimon.insert(corpus))

if __name__ == "__main__":
    # with open("./book.txt") as f:
    #     doc = f.read()

    asyncio.run(digimon.insert(corpus))

    a = asyncio.run(digimon.query("What is Ed Wood?"))
    # asyncio.run(digimon.query("Who is Scrooge?"))
    print(a)
