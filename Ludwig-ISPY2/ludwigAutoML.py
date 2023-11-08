import pandas
import logging
import pprint
import yaml
import pandas as pd
from ludwig.automl import auto_train
from ludwig.automl import create_auto_config
data = pd.read_csv("Paclitaxel_scaled_dataset.tsv", delimiter='\t')
config = yaml.safe_load(open("Paclitaxel_final_config.yaml"))
output = "Paclitaxel_auto_config.yaml"

auto_config = auto_train(
    dataset=data,
    target='TxResponse',
    time_limit_s=7200,
    tune_for_memory=False
)

with open(output, "w") as output_file:
    yaml.dump(auto_config, output_file)