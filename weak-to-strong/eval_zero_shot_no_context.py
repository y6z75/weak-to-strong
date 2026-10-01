import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "gpt2"
    print(f"Loading unadapted pretrained model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading SciQ dataset via allenai/sciq (test split)...")
    try:
        dataset = load_dataset("allenai/sciq", split="test")
    except Exception:
        dataset = load_dataset("allenai/sciq", "default", split="test")

    correct = 0
    total = 0

    print("Evaluating closed-book zero-shot accuracy (NO support context)...")
    with torch.no_grad():
        for example in tqdm(dataset):
            question = example["question"].strip()
            correct_ans = example["correct_answer"].strip()
            distractor = example["distractor1"].strip()

            # Closed-book prompt: Question and Answer token only
            prompt = f"Question: {question}\nAnswer: "

            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            target_ids = [tokenizer.encode(" " + ans, return_tensors="pt").to(device) for ans in [correct_ans, distractor]]

            candidate_losses = []
            for t_ids in target_ids:
                input_ids = torch.cat([prompt_ids, t_ids], dim=1)
                labels = input_ids.clone()
                labels[:, :prompt_ids.shape[1]] = -100

                outputs = model(input_ids, labels=labels)
                candidate_losses.append(outputs.loss.item())

            pred_idx = candidate_losses.index(min(candidate_losses))
            if pred_idx == 0:
                correct += 1
            total += 1

    acc = (correct / total) * 100
    print("\n" + "=" * 65)
    print(f"Closed-Book Zero-Shot GPT-2 Base (No Support): {acc:.2f}% ({correct}/{total})")
    print("=" * 65)

if __name__ == "__main__":
    main()