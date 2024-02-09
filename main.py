from logging import getLogger, basicConfig, INFO
from datasets.data_loader import CustomDataset

from generics.configs import DataConfig
basicConfig(level=INFO)

def main():
    logger = getLogger('main')
    config = DataConfig() 
    dataset = CustomDataset(config=config, subset_sample_count=100, mode="train")
    dataset.print_summary()



if __name__ == "__main__":
    main()
