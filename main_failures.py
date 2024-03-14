from datasets import data_loader_configs
from models import CustomModel, custom_model_configs
from utils.failure_utils import run_failure_analysis


def main():
    model_config = custom_model_configs.EfficientNetB0ConfigV1
    model = CustomModel.CustomModel(model_config)

    run_failure_analysis(
        model_config=model_config,
        data_loader_config=data_loader_configs.SmallBaseConfig,
        model=model,
    )


if __name__ == "__main__":
    main()
