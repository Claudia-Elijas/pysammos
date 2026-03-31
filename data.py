from pathlib import Path
import pooch

# Where data will be cached/expected
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "data_examples"

DATASET = pooch.create(
    path=EXAMPLES_DIR,
    base_url="https://zenodo.org/records/19351802/files/",
    registry={
        "data_examples.zip": "sha256:22718c1889ede9db5e01ae6834977c04a31566c69a455627e24a1a9708bef2b4",
    }
)

def fetch_example_data(force_download=False):
    """
    Fetch example data. If already present, skip download.
    
    Parameters
    ----------
    force_download : bool
        Force re-download even if data already exists.
    """
    if EXAMPLES_DIR.exists() and not force_download:
        print(f"Data already found at {EXAMPLES_DIR}, skipping download.")
        return EXAMPLES_DIR
    
    print("Downloading example data...")
    DATASET.fetch("data_examples.zip", processor=pooch.Unzip())
    return EXAMPLES_DIR