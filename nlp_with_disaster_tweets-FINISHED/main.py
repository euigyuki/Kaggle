from helper import load_glove_embeddings, train_model, evaluate_model, create_submission, load_data, prepare_datasets, create_data_loaders, setup_model
from config import config
import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    train_data, test_data = load_data()
    train_dataset, val_dataset, vocab = prepare_datasets(train_data)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_data, vocab)

    vocab_size = len(vocab)

    model = setup_model(len(vocab),vocab)
    best_model = train_model(train_loader, val_loader, model)
    print()
    print(f"Best validation accuracy: {best_model.best_val_accuracy:.4f}")
    print(f"Best validation loss: {best_model.best_val_loss:.4f}")
    predictions = evaluate_model(test_loader, best_model)
    create_submission(predictions, test_data, 'final_submission.csv')
    evaluate_model(test_loader, best_model)

if __name__ == '__main__':
    main()