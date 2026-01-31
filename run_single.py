
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from comer.datamodule import vocab
from comer.lit_comer import LitCoMER


def run_single(image_path: str, ckpt: str, device: str = "cuda:0") -> str:
    model = LitCoMER.load_from_checkpoint(ckpt, map_location=device)
    model.eval()
    model = model.to(device)

    img = Image.open(image_path)
    img_tensor = ToTensor()(img).to(device)
    mask = torch.zeros_like(img_tensor, dtype=torch.bool)

    with torch.no_grad():
        hyp = model.approximate_joint_search(img_tensor.unsqueeze(0), mask)[0]

    pred_indices = hyp.seq
    words = [
        vocab.idx2word[i]
        for i in pred_indices
        if i not in (vocab.PAD_IDX, vocab.SOS_IDX, vocab.EOS_IDX)
    ]
    return "".join(words)


if __name__ == "__main__":
    image_path = "comer_data/data/val/img/val000001.bmp"
    ckpt = "lightning_logs/version_48/checkpoints/epoch=259-step=61880-val_ExpRate=0.4218.ckpt"
    device = "cuda:0"

    print(f"Image: {image_path}")
    print(f"Checkpoint: {ckpt}")
    pred = run_single(image_path, ckpt, device)
    print(f"Prediction: {pred}")
