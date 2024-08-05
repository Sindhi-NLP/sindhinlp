import pandas as pd

def load_dataset(dataset_type: str = 'article_classification_dataset') -> pd.DataFrame:
    """
    Loads a specified dataset and returns it as a pandas DataFrame.
    
    Parameters:
    dataset_type (str): The type of dataset to load. Default is 'article_classification_dataset'.
    
    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    
    Raises:
    ValueError: If the dataset_type is not recognized.
    FileNotFoundError: If the dataset file does not exist.
    """

    dataset_paths = {
        'article_classification_dataset': 'sindhinlp/data/add.csv',
        'article_dataset': 'sindhinlp/data/dkad.csv'
    }
    
    if dataset_type not in dataset_paths:
        raise ValueError(f"Unrecognized dataset_type: {dataset_type}. Available types: {list(dataset_paths.keys())}")
    
    file_path = dataset_paths[dataset_type]
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    return df
