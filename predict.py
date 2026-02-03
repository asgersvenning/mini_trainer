import os
import urllib.parse
import urllib.request

from tqdm.auto import tqdm

try:
    from mini_trainer.hierarchical.predict import cli as default_args
    from mini_trainer.predict import main as mt_predict
except ImportError as e:
    e.add_note("`mini_trainer` does not seem to be installed, try `pip install -e .`.")
    raise e



def download(url, dest=None):
    if not dest:
        dest = os.path.basename(urllib.parse.urlparse(url).path)
        if not dest:
            raise ValueError("Cannot determine filename from URL; please provide a destination.")
            
    tmp = dest + ".tmp"
    
    try:
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
            total = int(r.headers.get('Content-Length', 0))
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=f'Downloading "{dest}"') as bar:
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

WEIGHT_VERSION = "hierarchical_model_v0.pt"
WEIGHT_SRC = f"https://anon.erda.au.dk/share_redirect/HE90eyuZCT/UKDK/{WEIGHT_VERSION}"


def run():
    args = default_args()

    if args["weights"] is None:
        if not os.path.exists(WEIGHT_VERSION):
            download(WEIGHT_SRC, WEIGHT_VERSION)
        args["weights"] = WEIGHT_VERSION
    if args["name"] in [None, "predict"]:
        args["name"] = "results"
    
    mt_predict(**args)


if __name__ == "__main__":
    run()