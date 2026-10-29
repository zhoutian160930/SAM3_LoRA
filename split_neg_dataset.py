#!/usr/bin/env python3
"""把「正常标注 + 错误标注」数据集按比例（默认 7:3）划分为 train / valid。

划分单位是**原图**：同一张图的 normal / wrong 两份标注永远落在同一个 split，
保证「图片一致」的配对关系不被破坏。

输入目录（data_dir）：
    normal_ann.json    正确标注
    wrong_ann.json     错误标注
    images/...         图片（file_name 相对 data_dir）

输出（out_dir）：
    train/
        _annotations.coco.json   # 正确标注（子集）
        wrong_ann.json           # 错误标注（同一批图片）
        images/...
    valid/  同上
    split_info.json              # seed、比例、划分清单

用法：
    python split_neg_dataset.py \
        --data_dir /root/work/六婆_负样本增广合并版_20261029_084208/sample_200 \
        --out_dir  /root/work/六婆_负样本增广合并版_20261029_084208/sample_200_split \
        --ratio 0.7 --seed 42 --link
"""

import argparse
import json
import random
from pathlib import Path

from prepare_neg_data import build_split, load, print_result, subset_coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="含 normal_ann.json / wrong_ann.json / images 的目录")
    ap.add_argument("--out_dir", required=True, help="输出根目录（会生成 train/ 和 valid/）")
    ap.add_argument("--normal_json", default="normal_ann.json")
    ap.add_argument("--wrong_json", default="wrong_ann.json")
    ap.add_argument("--images_dir", default="images")
    ap.add_argument("--ratio", type=float, default=0.7, help="训练集比例，默认 0.7")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="图片用软链接（省空间）")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    normal = load(data_dir / args.normal_json)
    wrong = load(data_dir / args.wrong_json)

    names = sorted(im["file_name"] for im in normal["images"])
    if len(names) < 2:
        raise SystemExit(f"图片太少（{len(names)}），无法划分")

    rng = random.Random(args.seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * args.ratio))
    n_train = max(1, min(len(shuffled) - 1, n_train))
    train_names = sorted(shuffled[:n_train])
    valid_names = sorted(shuffled[n_train:])

    assert not (set(train_names) & set(valid_names))
    assert len(train_names) + len(valid_names) == len(names)

    print(f"总图片 {len(names)} 张，按 {args.ratio:.0%}/{(1-args.ratio):.0%} 划分"
          f"（seed={args.seed}）")
    print(f"  train: {len(train_names)} 张   valid: {len(valid_names)} 张\n")

    splits = {}
    for split, keep in [("train", train_names), ("valid", valid_names)]:
        n_sub = subset_coco(normal, keep)
        w_sub = subset_coco(wrong, keep)
        stats = build_split(n_sub, w_sub, data_dir / args.images_dir,
                            out_dir / split, link=args.link)
        print_result(stats, link=args.link)
        splits[split] = {
            "images": len(keep),
            "normal_instances": len(n_sub["annotations"]),
            "wrong_instances": len(w_sub["annotations"]),
            "files": keep,
        }
        print()

    info = {
        "source_data_dir": str(data_dir),
        "ratio": args.ratio,
        "seed": args.seed,
        "total_images": len(names),
        "splits": splits,
        "note": "同一张原图的 normal/wrong 标注始终在同一 split，两份 json 的图片列表一致",
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"划分清单: {out_dir / 'split_info.json'}")
    print("\n训练配置：")
    print(f"training:")
    print(f"  data_dir: \"{out_dir}\"")
    print("negative_supervision:")
    print("  enable: true")
    print("  ann_file: \"wrong_ann.json\"")


if __name__ == "__main__":
    main()
