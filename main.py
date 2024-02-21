from utils.failure_utils import failure_analysis
from generics import Paths



def main():

    model_config = None 
    data_loader_config = None 
    model = None # base model to load chkpts in 

    model_dirs = [f"""{Paths.BEST_MODEL_CHECKPOINTS}
                  /best_{model_config.MODEL}{model_config.NAME}{data_loader_config.NAME}
                  fold{fold}.pth""" for fold in range(model_config.FOLDS)]
    
    for fold in range(model_config.FOLDS):
        dataset = None # how to get the right val fold 
        failure_analysis(model_dir=model_dirs[fold], model=model, dataset=dataset)
    
    

if __name__ == "__main__":
    main()