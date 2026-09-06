from pathlib import Path

from configs import config


def test_project_root_exists():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.PROJECT_ROOT.exists()


def test_dataset_paths_are_path_objects():
    assert isinstance(config.DATASET_ROOT, Path)
    assert isinstance(config.SAMPLE_VIDEO_ROOT, Path)
    assert isinstance(config.OUTPUT_ROOT, Path)
    assert isinstance(config.SAVED_MODEL_ROOT, Path)


def test_output_directories_exist():
    assert config.OUTPUT_ROOT.exists()
    assert config.SAVED_MODEL_ROOT.exists()


def test_feature_dimensions():
    assert config.LIP_FEATURE_SIZE == 80
    assert config.LIP_VELOCITY_SIZE == 80
    assert config.MFCC_FEATURE_SIZE == 39

    expected_size = (
        config.LIP_FEATURE_SIZE
        + config.LIP_VELOCITY_SIZE
        + config.MFCC_FEATURE_SIZE
    )

    assert config.FEATURE_SIZE == expected_size


def test_dataset_split_ratios():
    total = (
        config.TRAIN_RATIO
        + config.VALIDATION_RATIO
        + config.TEST_RATIO
    )

    assert total == 1.0


def test_training_configuration():
    assert config.BATCH_SIZE > 0
    assert config.EPOCHS > 0
    assert config.LEARNING_RATE > 0
    assert config.RANDOM_SEED >= 0
