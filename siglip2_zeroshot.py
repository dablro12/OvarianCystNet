import os
from transformers import pipeline
import torch
import pandas as pd

def zero_shot_classification(
    img_root_dir,
    data_path,
    save_root_dir,
    model_ckpt="google/siglip2-so400m-patch14-384",
    batch_size=16,
    labels=None
):
    """
    Perform zero-shot image classification and save the top-1 results as a CSV.
    """
    if labels is None:
        labels = ["Benign", "Borderline", "Malignant"]
    label_to_num = {label: i for i, label in enumerate(labels)}

    os.makedirs(save_root_dir, exist_ok=True)

    # Set up the Huggingface pipeline
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        model=model_ckpt,
        task="zero-shot-image-classification",
        device=device
    )

    # Load data
    data_df = pd.read_csv(data_path)
    img_files = data_df['filename'].astype(str).tolist()
    gt_labels = data_df['USG_Ontology'].tolist()

    top1_results = {
        "filename": [],
        "label": [],
        "preds": [],
        "probs": [],
    }

    # Inference in batches
    for start_idx in range(0, len(img_files), batch_size):
        batch_imgs = img_files[start_idx : start_idx + batch_size]
        batch_img_paths = [os.path.join(img_root_dir, f"{img_file}.png") for img_file in batch_imgs]

        outputs = classifier(batch_img_paths, candidate_labels=labels)
        for i, out in enumerate(outputs):
            # Get the highest scoring label
            top_result = max(out, key=lambda x: x["score"])
            top1_results["filename"].append(batch_imgs[i])
            top1_results["label"].append(gt_labels[start_idx + i])
            top1_results["preds"].append(label_to_num[top_result["label"]])
            top1_results["probs"].append(top_result["score"])

    # Ensure lengths are consistent before saving
    min_len = min(
        len(top1_results["filename"]),
        len(top1_results["label"]),
        len(top1_results["preds"]),
        len(top1_results["probs"])
    )
    df = pd.DataFrame({k: v[:min_len] for k, v in top1_results.items()})
    csv_path = os.path.join(save_root_dir, "top1_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to {csv_path}")

if __name__ == "__main__":
    # 사용 예시
    IMG_ROOT_DIR = "/workspace/pcos_dataset/Dataset"
    DATA_PATH = "/workspace/pcos_dataset/labels/통합_Dataset_info.csv"
    SAVE_ROOT_DIR = f"/workspace/pcos_dataset/results/zero_shot/siglip2"

    for CKPT in ["google/siglip2-base-patch32-256", "google/siglip2-base-patch16-384", "google/siglip2-large-patch16-384", "google/siglip2-so400m-patch14-384", "google/siglip2-so400m-patch16-384"]:
        zero_shot_classification(
            img_root_dir=IMG_ROOT_DIR,
            data_path=DATA_PATH,
            save_root_dir=f"{SAVE_ROOT_DIR}/{CKPT}",
            model_ckpt=CKPT,
            batch_size=16,
            labels=["Benign", "Borderline", "Malignant"]
        )