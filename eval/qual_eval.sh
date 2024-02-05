#!/bin/bash

ckpt_paths=(
    # "checkpoint/cifar100_baseline.ckpt" # baseline cifar100
    "checkpoint/helpful-haze-123.ckpt" # baseline ImageNetS50
    # "checkpoint/lico_copper_wood.ckpt" # lico cifar100
    "checkpoint/default-LICO-210.ckpt" # lico ImageNetS50
)

save_outputs=(
    # "saliency_output_baseline_cifar100"
    "saliency_output_baseline_ImageNetS50"
    # "saliency_output_lico_cifar100"
    "saliency_output_lico_ImageNetS50"
)

training_methods=(
  "baseline"
  "lico"
)
datasets=(
  # "cifar100"
  "ImageNetS50"
)
img_data_paths=(
    # "data/cifar100/val"
    "data/ImageNetS50/validation"
)

# Loop over combinations of training methods and datasets
for ((i=0; i<${#training_methods[@]}; i++)); do
    for ((j=0; j<${#datasets[@]}; j++)); do
        index=$((i * ${#datasets[@]} + j))

        # Extracting variables
        ckpt_path=${ckpt_paths[$index]}
        save_output=${save_outputs[$index]}
        training_method=${training_methods[$i]}
        dataset=${datasets[$j]}
        img_data=${img_data_paths[$j]}

        # Print configuration
        echo "Running configuration:"
        echo "  Training Method: $training_method"
        echo "  Dataset: $dataset"
        echo "  Checkpoint Path: $ckpt_path"
        echo "  Save Output: $save_output"
        echo "  Img Data Path: $img_data"

        # Constructing the Python command with variables
        python -m eval.generate_heatmaps \
            --ckpt-path "$ckpt_path" \
            --save-output "$save_output" \
            --training-method "$training_method" \
            --dataset "$dataset" \
            --img-data "$img_data"
    done
done
