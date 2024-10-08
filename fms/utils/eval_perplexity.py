import torch
import torch.nn as nn
from tqdm import tqdm
# from lm_eval import evaluator
from datasets import load_dataset
# from transformers import pipeline
# from evaluate import load as load_metric
# from lm_eval.tasks import initialize_tasks
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.models.whisper.english_normalizer import BasicTextNormalizer


# def get_device():
#     if torch.backends.mps.is_available():
#         return "mps"
#     elif torch.cuda.is_available():
#         return "cuda:0"
#     else:
#         return "cpu"


def evaluate_perplexity(model, tokenizer, device="cuda"):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer.tokenize("\n\n".join(data["text"]))
    data = tokenizer.convert_tokens_to_ids(data)
    data = torch.tensor(data, dtype=torch.long, device=device).unsqueeze(0)

    # breakpoint()

    seqlen = 2048
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(device)
            with torch.no_grad():
                logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()
