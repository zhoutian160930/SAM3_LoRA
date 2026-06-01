"""
将 LoRA 微调权重合并到原始 SAM3 模型权重中。

原理：
  LoRA 的 forward:  lora_out = x @ lora_A @ lora_B * scaling
  对于 Linear 层 W (out_features, in_features):
    delta_W = lora_B.T @ lora_A.T * scaling
    W_merged = W + delta_W

键名映射：
  - LoRA 中 q_proj / k_proj / v_proj (分开的) → base 中 in_proj_weight (合并的 QKV)
  - LoRA 中 out_proj / linear1 / linear2 / hs_proj / prompt_proj → base 中对应 .weight
  - base 模型键名有 'detector.' 前缀，LoRA 键名没有
"""

import torch
import argparse
from pathlib import Path


def merge_lora_into_base(
    base_path: str,
    lora_path: str,
    output_path: str,
    rank: int = 32,
    alpha: int = 64,
):
    scaling = alpha / rank
    print(f"rank={rank}, alpha={alpha}, scaling={scaling:.4f}")

    print(f"\n加载 base 模型: {base_path}")
    base_sd = torch.load(base_path, map_location="cpu")
    print(f"  base 模型键数量: {len(base_sd)}")

    print(f"\n加载 LoRA 权重: {lora_path}")
    lora_sd = torch.load(lora_path, map_location="cpu")
    print(f"  LoRA 键数量: {len(lora_sd)}")

    # 将 lora 权重按模块分组: {module_path: {'lora_A': tensor, 'lora_B': tensor}}
    lora_modules = {}
    for key, val in lora_sd.items():
        # key 形如: transformer.decoder.layers.0.cross_attn.q_proj.lora.lora_A
        assert key.endswith(".lora_A") or key.endswith(".lora_B"), f"未知 LoRA key 格式: {key}"
        if key.endswith(".lora_A"):
            module_path = key[: -len(".lora.lora_A")]
        else:
            module_path = key[: -len(".lora.lora_B")]
        if module_path not in lora_modules:
            lora_modules[module_path] = {}
        if key.endswith(".lora_A"):
            lora_modules[module_path]["lora_A"] = val
        else:
            lora_modules[module_path]["lora_B"] = val

    print(f"\n  LoRA 模块数量: {len(lora_modules)}")

    merged_count = 0
    skipped = []

    for module_path, weights in lora_modules.items():
        lora_A = weights["lora_A"]  # (in_features, rank)
        lora_B = weights["lora_B"]  # (rank, out_features)

        # delta_W shape: (out_features, in_features)
        delta_W = (lora_B.T @ lora_A.T) * scaling

        # 判断该 LoRA 模块对应 base 中哪个权重 key
        # 规则1: q_proj / k_proj / v_proj → in_proj_weight (分 slice)
        if module_path.endswith(".q_proj"):
            base_prefix = "detector." + module_path[: -len(".q_proj")]
            base_key = base_prefix + ".in_proj_weight"
            slice_idx = 0
        elif module_path.endswith(".k_proj"):
            base_prefix = "detector." + module_path[: -len(".k_proj")]
            base_key = base_prefix + ".in_proj_weight"
            slice_idx = 1
        elif module_path.endswith(".v_proj"):
            base_prefix = "detector." + module_path[: -len(".v_proj")]
            base_key = base_prefix + ".in_proj_weight"
            slice_idx = 2
        else:
            # out_proj / linear1 / linear2 / hs_proj / prompt_proj 等直接映射
            base_key = "detector." + module_path + ".weight"
            slice_idx = None

        if base_key not in base_sd:
            skipped.append((module_path, base_key))
            continue

        if slice_idx is not None:
            # in_proj_weight 形状: (3 * embed_dim, embed_dim)
            embed_dim = delta_W.shape[0]
            full_embed = base_sd[base_key].shape[0]
            assert full_embed == 3 * embed_dim, (
                f"in_proj_weight 形状不匹配: {base_sd[base_key].shape}, embed_dim={embed_dim}"
            )
            start = slice_idx * embed_dim
            end = start + embed_dim
            base_sd[base_key][start:end] += delta_W
        else:
            assert base_sd[base_key].shape == delta_W.shape, (
                f"形状不匹配: {base_key} {base_sd[base_key].shape} vs delta {delta_W.shape}"
            )
            base_sd[base_key] += delta_W

        merged_count += 1

    print(f"\n  成功合并: {merged_count} 个模块")
    if skipped:
        print(f"  跳过 (base 中未找到对应 key): {len(skipped)} 个模块")
        for mp, bk in skipped:
            print(f"    LoRA: {mp}  →  base key: {bk}")

    print(f"\n保存合并后的模型到: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_sd, output_path)
    print("✅ 合并完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 LoRA 权重合并到 SAM3 base 模型")
    parser.add_argument(
        "--base", 
        default="/home/model/sam3_pth/sam3pt/sam3.pt",
        help="原始 SAM3 base 模型路径"
    )
    parser.add_argument(
        "--lora",
        default="/home/model/work/SAM3_LoRA/outputs/sam3_lora_full_1/best_lora_weights.pt",
        help="LoRA 权重文件路径"
    )
    parser.add_argument(
        "--output",
        default="/home/model/sam3_pth/finetuning_weights/duidie_weights/pytorch/sam3_merged.pt",
        help="合并后模型的保存路径"
    )
    parser.add_argument("--rank",  type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    args = parser.parse_args()

    merge_lora_into_base(
        base_path=args.base,
        lora_path=args.lora,
        output_path=args.output,
        rank=args.rank,
        alpha=args.alpha,
    )
