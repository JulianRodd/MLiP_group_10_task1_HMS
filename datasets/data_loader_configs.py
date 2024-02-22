class BaseDataConfig:
    NAME = "BaseDataConfig"
    # Common parameters across all configs
    ONE_CROP_PER_PERSON = True
    USE_WAVELET: bool = None
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    SHUFFLE_TRAIN = True
    NUM_WORKERS = 0
    PIN_MEMORY = True
    DROP_LAST = True
    NAMES = ["LL", "LP", "RP", "RR"]
    FEATS = [
        ["Fp1", "F7", "T3", "T5", "O1"],
        ["Fp1", "F3", "C3", "P3", "O1"],
        ["Fp2", "F8", "T4", "T6", "O2"],
        ["Fp2", "F4", "C4", "P4", "O2"],
    ]
    VAL_SPLIT_RATIO = 0.2
    SUBSET_SAMPLE_COUNT: int = 0
    N_COMPONENTS = 4
    SUBSET_SAMPLE_COUNT = 0
    USE_PRELOADED_EEG_SPECTROGRAMS = True
    USE_PRELOADED_SPECTROGRAMS = True
    APPLY_ICA_EEG_SPECTROGRAMS = False
    NORMALIZE_EEG_SPECTROGRAMS = False
    NORMALIZE_RAW_EEG = False
    NORMALIZE_INDIVIDUALLY = True
    APPLY_MSPCA_RAW_EEG = False
    APPLY_ICA_RAW_EEG = False
    APPLY_MSPCA_EEG_SPECTROGRAMS = False
    
    def __init__(self):
        super().__init__()
        # Constraint checks for instance creation
        self.check_constraints()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

        # Constraint checks for subclass creation
        cls.check_constraints()

    @classmethod
    def check_constraints(cls):
        # Check if RAW processing and corresponding Spectrogram processing are both enabled
        if getattr(cls, 'APPLY_MSPCA_RAW_EEG', False) and getattr(cls, 'APPLY_MSPCA_EEG_SPECTROGRAMS', False):
            raise ValueError(f"{cls.NAME}: Both APPLY_MSPCA_RAW_EEG and APPLY_MSPCA_EEG_SPECTROGRAMS cannot be True at the same time.")

        if getattr(cls, 'APPLY_ICA_RAW_EEG', False) and getattr(cls, 'APPLY_ICA_EEG_SPECTROGRAMS', False):
            raise ValueError(f"{cls.NAME}: Both APPLY_ICA_RAW_EEG and APPLY_ICA_EEG_SPECTROGRAMS cannot be True at the same time.")

        # Check: If APPLY_MSPCA_RAW_EEG or APPLY_ICA_RAW_EEG is True, then USE_PRELOADED_EEG_SPECTROGRAMS must be False
        if (getattr(cls, 'APPLY_MSPCA_RAW_EEG', False) or getattr(cls, 'APPLY_ICA_RAW_EEG', False)) and getattr(cls, 'USE_PRELOADED_EEG_SPECTROGRAMS', False):
            raise ValueError(f"{cls.NAME}: USE_PRELOADED_EEG_SPECTROGRAMS must be False when APPLY_MSPCA_RAW_EEG or APPLY_ICA_RAW_EEG is True.")

class ExtremelySmallBaseConfig(BaseDataConfig):
    BATCH_SIZE_TRAIN = 2
    BATCH_SIZE_TEST = 1
    BATCH_SIZE_VAL = 1
    SUBSET_SAMPLE_COUNT = 10
    
class SmallBaseConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 100
    BATCH_SIZE_TEST = 4
    BATCH_SIZE_VAL = 4
    BATCH_SIZE_TRAIN = 8

# Renamed Config classes
class SmallConfig_ICA_Normalize(SmallBaseConfig):
    APPLY_ICA_EEG_SPECTROGRAMS = True
    NORMALIZE_EEG_SPECTROGRAMS = True

class SmallConfig_MSPCA(SmallBaseConfig):
    APPLY_MSPCA_EEG_SPECTROGRAMS = True

class SmallConfig_Normalize_MSPCA(SmallBaseConfig):
    NORMALIZE_EEG_SPECTROGRAMS = True
    APPLY_MSPCA_EEG_SPECTROGRAMS = True

class SmallConfig_ICA_MSPCA(SmallBaseConfig):
    APPLY_ICA_EEG_SPECTROGRAMS = True
    APPLY_MSPCA_EEG_SPECTROGRAMS = True

class SmallConfig_AllFeatures(SmallBaseConfig):
    APPLY_ICA_EEG_SPECTROGRAMS = True
    NORMALIZE_EEG_SPECTROGRAMS = True
    APPLY_MSPCA_EEG_SPECTROGRAMS = True

class SmallConfig_Normalize(SmallBaseConfig):
    NORMALIZE_EEG_SPECTROGRAMS = True

class SmallConfig_EnhancedICA(SmallBaseConfig):
    N_COMPONENTS = 5
    APPLY_ICA_EEG_SPECTROGRAMS = True

class SmallConfig_EnhancedMSPCA(SmallBaseConfig):
    N_COMPONENTS = 6
    APPLY_MSPCA_EEG_SPECTROGRAMS = True

class SmallConfig_ICA(SmallBaseConfig):
    APPLY_ICA_EEG_SPECTROGRAMS = True

class SmallConfig_MSPCA_Enhanced(SmallBaseConfig):
    N_COMPONENTS = 7
    APPLY_MSPCA_EEG_SPECTROGRAMS = True

class ThousandSamplesBaseConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 1000
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    BATCH_SIZE_TRAIN = 32
class ThousandSamplesBaseConfig_Normalize(ThousandSamplesBaseConfig):
    NORMALIZE_EEG_SPECTROGRAMS = True
    
    
class ThousandSamplesBaseConfig_EnhancedMSPCA(ThousandSamplesBaseConfig):
    N_COMPONENTS = 5
    APPLY_MSPCA_EEG_SPECTROGRAMS = True

class SmallConfig_Normalize_Group(SmallBaseConfig):
    NORMALIZE_EEG_SPECTROGRAMS = True
    NORMALIZE_INDIVIDUALLY = False

class SmallConfig_Normalize_Group_Raw_MSPCA(SmallBaseConfig):
    NORMALIZE_EEG_SPECTROGRAMS = True
    NORMALIZE_INDIVIDUALLY = False
    APPLY_MSPCA_RAW_EEG = True
    USE_PRELOADED_EEG_SPECTROGRAMS = False

class SmallConfig_Normalize_Group_Raw_ICA(SmallBaseConfig):
    NORMALIZE_EEG_SPECTROGRAMS = True
    NORMALIZE_INDIVIDUALLY = False
    APPLY_ICA_RAW_EEG = True
    USE_PRELOADED_EEG_SPECTROGRAMS = False

class SmallConfig_Raw_MSPCA(SmallBaseConfig):
    APPLY_MSPCA_RAW_EEG = True
    USE_PRELOADED_EEG_SPECTROGRAMS = False

class SmallConfig_Raw_ICA(SmallBaseConfig):
    APPLY_ICA_RAW_EEG = True
    USE_PRELOADED_EEG_SPECTROGRAMS = False
    
class Config_Normalize_Group_Raw_ICA(BaseDataConfig):
    NORMALIZE_EEG_SPECTROGRAMS = True
    NORMALIZE_INDIVIDUALLY = False
    APPLY_ICA_RAW_EEG = True
    USE_PRELOADED_EEG_SPECTROGRAMS = False

 # SmallBaseConfig, SmallConfig_Normalize, SmallConfig_EnhancedMSPCA,SmallConfig_Raw_ICA,  SmallConfig_Normalize_Indiv, SmallConfig_Normalize_Indiv_Raw_MSPCA, SmallConfig_Normalize_Indiv_Raw_ICA, SmallConfig_Normalize_Indiv_Raw_MSPCA, SmallConfig_Normalize_Indiv_Raw_ICA, 
DATASET_GRID_SEARCH = [SmallConfig_Normalize_Group_Raw_ICA]
