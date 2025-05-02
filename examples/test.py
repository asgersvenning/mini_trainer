import json

import numpy as np
import torch
from PIL import Image

from mini_trainer.builders import base_model_builder

if __name__ == "__main__":
    dev, dt = torch.device("cuda:0"), torch.float32
    paths = [
        '/home/asger/mini_trainer/examples/hierarchical/gmo_traits/7762841/e4c0f18516f8fd50e3989b10fec5e9d85dcafa57.jpg',
        '/home/asger/mini_trainer/examples/hierarchical/gmo_traits/1938810/70668f47cf77c6ad67e6afdb1031d02605743a2c.jpg',
        '/home/asger/mini_trainer/examples/hierarchical/gmo_traits/5135513/33138d8c013adab840f31b8c1a7ae9be3f648fde.jpg',
        '/home/asger/mini_trainer/examples/hierarchical/gmo_traits/1926550/d29dfedcc6ebffadda1ebc330de2d9c771c0e4f2.jpg',
        '/home/asger/mini_trainer/examples/hierarchical/gmo_traits/1918795/6a65736ba9050a2afa43a82c14513bc5512d53b2.jpg'
    ]

    with open("quality/class_description.json", "rb") as f:
        cls2desc = {k : v.strip().split(":")[0] for k, v in json.load(f).items()}

    with open("quality/class_index.json", "rb") as f:
        cls2idx = json.load(f)
    idx2cls = {v : k for k, v in cls2idx.items()}

    qcmodel, mproc = base_model_builder(
        model="efficientnet_v2_s",
        weights="quality/efficientnet_v2_s_full_e15.pt",
        device=dev,
        dtype=dt,
        fine_tune=False,
        num_classes=len(cls2desc)
    )
    qcmodel.eval()

    def load_image(path):
        return mproc(torch.tensor(np.array(Image.open(path)).copy()).permute(2,0,1).float() / 255.)

    with torch.inference_mode():
        with torch.autocast(dev.type, dt):
            tout = qcmodel(torch.stack([load_image(p) for p in paths]).to(dev, dt)).cpu()
        tconfs = tout.softmax(1)
        tconf, tpred = tconfs.max(1).values, tout.argmax(1)
        tcls = [idx2cls[i.item()] for i in tpred]

    for i in range(5):
        print("_"*60)
        print(f"PREDICTION #{i}")
        print(f"Model output: [{', '.join([f'{v:.1f}' for v in tout[i]])}]")
        print(f"Model confidences: [{', '.join([f'{c.item():.0%}' for c in tconfs[i]])}]")
        print(f"Model prediction & confidence: {cls2desc[tcls[i]]} ({tconf[i]:.1%})")
        print("#"*60)
        print()