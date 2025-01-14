import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# def evaluate(model_checkpoint: str = 'models/model.pth') -> None:
#     """Evaluate a trained model."""
#     print("Evaluating like my life depended on it")
#     print(model_checkpoint)

#     model = MyAwesomeModel().to(DEVICE)
#     model.load_state_dict(torch.load(model_checkpoint))

#     _, test_set = corrupt_mnist()
#     test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

#     model.eval()
#     correct, total = 0, 0
#     for img, target in test_dataloader:
#         img, target = img.to(DEVICE), target.to(DEVICE)
#         y_pred = model(img)
#         correct += (y_pred.argmax(dim=1) == target).float().sum().item()
#         total += target.size(0)
#     print(f"Test accuracy: {correct / total}")



def evaluate_translation(model_checkpoint: str = 'models/model.pth', test_dataset=None) -> None:
    """
    Evaluate a trained mBERT model on translation from English to French using BLEU score.
    Assumes the input data is already tokenized.
    """
    print("Evaluating translation model...")
    print(f"Loading model from: {model_checkpoint}")

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained("path/to/your/model")
    model.to(DEVICE)

    if test_dataset is None:
        raise ValueError("You need to provide a test dataset for evaluation.")

    test_dataloader = DataLoader(test_dataset, batch_size=32)

    model.eval()
    references = []
    hypotheses = []

    for batch in test_dataloader:
        inputs, targets = batch['input_ids'], batch['target_ids']  # Assuming dataset has tokenized 'input_ids' and 'target_ids'
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_length=targets.size(1))
        
        # Decode tokens to strings (if needed for BLEU)
        predictions = outputs.cpu().tolist()
        references.extend([[t.tolist()] for t in targets])  # Wrap targets in a list for BLEU format
        hypotheses.extend(predictions)

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"BLEU score: {bleu_score:.4f}")




if __name__ == "__main__":
    typer.run(evaluate)