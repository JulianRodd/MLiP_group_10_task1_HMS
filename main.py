# from utils.failure_utils import run_failure_analysis
# from datasets import data_loader_configs
# from models import custom_model_configs, CustomModel
from utils.failure_utils import majority_confusion, avg_mispred, plot_avg_mispred
from generics import Generics, Paths
import numpy as np



def main():
    y_pred = np.array([[1,0.1,2,0.1, 0.1, 0.1], [1,0.1,2,0.1, 0.1, 0], [1,0.1,0.1,5, 0.1, 0.1], [1,2,0.1,0.1, 0.1, 0.1], [1,0.1,0.1,3, 0.1, 5], [1,0.1,0.1,3, 0.1, 5]])
    y_true = np.array([[1,0.1,0.1,0.1, 0.1, 0.1], [0.1,2,0.1,0.1,1, 0.1], [1,2,3,0.1, 0.1, 0.1], [1,0.1,2,4, 0.1, 0], [1,0.1,2,0.1, 5, 0], [1,0.1,2,0.1, 5, 6]])
    # print(majority_confusion(y_pred, y_true, Generics.LABEL_COLS))

    misp = avg_mispred(y_pred, y_true, by_class=True)
    print(misp)
    plot_avg_mispred(misp, by_class=True)

    # model_config = custom_model_configs.EfficientNetB0ConfigV1
    # model=CustomModel.CustomModel(model_config)

    # run_failure_analysis(model_config=model_config,
    #     data_loader_config=data_loader_configs.SmallBaseConfig, model=model)
    
    

if __name__ == "__main__":
    main()