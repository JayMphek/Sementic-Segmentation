class Config:
    # Dataset settings
    DATA_ROOT = '../IDDAW'  
    TRAIN_SPLIT = 'train'
    VAL_SPLIT = 'val'
    TARGET_SIZE = (512, 256)  

    # Dataset size limits
    TOTAL_SAMPLES = 1000  
    TRAIN_SPLIT_RATIO = 0.70  
    VAL_SPLIT_RATIO = 0.15    
    TEST_SPLIT_RATIO = 0.15   
    MAX_TRAIN_SAMPLES = int(TOTAL_SAMPLES * TRAIN_SPLIT_RATIO)
    MAX_VAL_SAMPLES = int(TOTAL_SAMPLES * VAL_SPLIT_RATIO)
    MAX_TEST_SAMPLES = int(TOTAL_SAMPLES * TEST_SPLIT_RATIO)
    
    # Model settings
    MODEL_NAME = 'segformer'  

    # SegFormer specific settings
    EMBED_DIMS = [64, 128, 320, 512]  
    NUM_HEADS = [1, 2, 5, 8]  
    MLP_RATIOS = [4, 4, 4, 4]  
    DEPTHS = [3, 4, 6, 3]  
    PATCH_SIZE = 4 
    DECODER_DIM = 256  
    
    # Training settings
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.09
    
    # Optimizer settings
    OPTIMIZER = 'adamw'  
    MOMENTUM = 0.9 
    BETAS = (0.9, 0.999) 

    # Learning rate scheduler
    SCHEDULER = 'cosine'  
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Loss function
    LOSS_FN = 'cross_entropy'  
    IGNORE_INDEX = 255  
    
    # Data augmentation
    USE_AUGMENTATION = True
    RANDOM_FLIP = 0.5
    RANDOM_CROP = False
    COLOR_JITTER = True
    
    # Training settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    GRADIENT_CLIP = 1.0
    
    # Checkpointing
    SAVE_DIR = './checkpoints'
    SAVE_FREQ = 5 
    BEST_MODEL_NAME = 'best_model.pth'
    LAST_MODEL_NAME = 'last_model.pth'
    
    # Logging
    LOG_INTERVAL = 10  
    USE_TENSORBOARD = False
    
    # Device
    DEVICE = 'cuda'  

    # Random seed for reproducibility
    SEED = 42
    
    # Resume training
    RESUME_FROM = None  
    
    @classmethod
    def display(cls):
        print("=" * 60)
        print("Configuration Settings")
        print("=" * 60)
        
        config_instance = cls()
        for key in dir(cls):
            if not key.startswith('_') and key != 'display':
                value = getattr(config_instance, key)
                if not callable(value):
                    print(f"{key:.<30} {value}")
        
        print("\nDataset Split (calculated):")
        print(f"{'Total samples':<30} {config_instance.TOTAL_SAMPLES}")
        print(f"{'Training samples (70%)':<30} {config_instance.MAX_TRAIN_SAMPLES}")
        print(f"{'Validation samples (15%)':<30} {config_instance.MAX_VAL_SAMPLES}")
        print(f"{'Test samples (15%)':<30} {config_instance.MAX_TEST_SAMPLES}")
        print("=" * 60)