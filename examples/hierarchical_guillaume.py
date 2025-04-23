import os, shutil

from hierarchical.base.integration import hierarchical_parse_class_index
from hierarchical.guillaume.setup import erda_to_combinations
from pyremotedata.implicit_mount import IOHandler, RemotePathIterator

from tqdm import tqdm as TQDM

SHARE_LINK = "JlICFo26h8"
IMAGE_DIR = os.path.abspath(os.path.join("hierarchical", "gmo_traits"))

if __name__ == "__main__":
    result = hierarchical_parse_class_index(
        dir = SHARE_LINK,
        dir2comb_fn=erda_to_combinations 
    )
    # if os.path.isdir(IMAGE_DIR):
    #     shutil.rmtree(IMAGE_DIR)
    # if os.path.exists(IMAGE_DIR):
    #     raise RuntimeError(f'Error while cleaning the image directory: {IMAGE_DIR}')
    os.makedirs(IMAGE_DIR, exist_ok=True)
    with IOHandler(user=SHARE_LINK, password=SHARE_LINK, remote="io.erda.au.dk", verbose=False) as io:
        ri = RemotePathIterator(io)
        ri.subset([i for i, rp in enumerate(ri.remote_paths) if not os.path.exists(os.path.join(IMAGE_DIR, *rp.split("/")))])
        for lp, rp in TQDM(ri, desc="Downloading files..."):
            dst = os.path.join(IMAGE_DIR, *rp.split("/"))
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            shutil.copy2(lp, dst)