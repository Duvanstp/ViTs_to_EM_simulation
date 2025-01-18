import os
import io
import requests
import numpy as np

def download_file(url, save_path):
    """
    Downloads a file from a URL and saves it to the specified path.
    """
    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"File saved to {save_path}.")

def ensure_data_exists(url, local_path):
    """
    Ensures the data file exists locally. If not, downloads it.
    """
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        download_file(url, local_path)

def load_data_from_local_or_remote(local_path, url):
    """
    Loads data from a local file or downloads it if not present.
    Returns the loaded numpy data.
    """
    ensure_data_exists(url, local_path)
    return np.load(local_path)

def load_train_data():
    """
    Loads train data, either locally or by downloading it.
    """
    local_path = "data/train_ds.npz"
    url = "http://metanet.stanford.edu/static/search/waveynet/data/train_ds.npz"
    train_data = load_data_from_local_or_remote(local_path, url)
    return train_data['structures'], train_data['Hy_fields'], train_data['dielectric_permittivities']

def load_test_data():
    """
    Loads test data, either locally or by downloading it.
    """
    local_path = "data/test_ds.npz"
    url = "http://metanet.stanford.edu/static/search/waveynet/data/test_ds.npz"
    test_data = load_data_from_local_or_remote(local_path, url)
    return (
        test_data['structures'],
        test_data['Hy_fields'],
        test_data['Ex_fields'],
        test_data['Ez_fields'],
        test_data['efficiencies'],
        test_data['dielectric_permittivities']
    )

if __name__ == "__main__":
    try:
        print("Loading train data...")
        train_structures, train_Hy_fields, train_dielectric_permittivities = load_train_data()
        print("Train data loaded successfully.")

        print("Loading test data...")
        test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = load_test_data()
        print("Test data loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
