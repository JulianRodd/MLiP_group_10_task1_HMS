from datasets.data_loader import CustomDataset
from generics.configs import DataConfig
from utils.general_utils import get_logger


def main():
    logger = get_logger('main')
    config = DataConfig() 
    dataset = CustomDataset(config=config, subset_sample_count=100, mode="train", cache=True)
    dataset.print_summary()
    dataset.plot_samples(n_samples=1)




if __name__ == "__main__":
    main()
