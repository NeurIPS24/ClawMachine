gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-clawmachine-id-dual-1e5"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /home/Clawmachine/evaluation/model_GND_fetch_mirage.py \
        --model-path /home/Clawmachine/checkpoints/$CKPT \
        --question-file ./playground/data/eval/GND_refcocos/Refcoco_Val_questions/${IDX}.json \
        --answers-file /home/Clawmachine/answer_file/GND/$CKPT/${IDX}_greedy.jsonl &
done

wait

output_file=/home/Clawmachine/answer_file/GND/$CKPT/merge_greedy_nqi.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /home/Clawmachine/answer_file/GND/$CKPT/${IDX}_greedy.jsonl >> "$output_file"
done

