import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import evaluate
import argparse
import os
import logging
from torch.optim.lr_scheduler import LambdaLR
from tatoebatools import tatoeba
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, num_classes, max_len, hidden_dim, vocab_size, num_layers=2):
        super().__init__()
        self.max_len = max_len
        self.num_classes = num_classes       
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # First conv layer: vocab_size -> hidden_dim
        self.conv1 = nn.Conv1d(self.vocab_size, self.hidden_dim, kernel_size=3, padding=1)
        
        # Intermediate conv layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2)
            for _ in range(num_layers)
        ])
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_onehot = F.one_hot(x, num_classes=self.vocab_size)
        x_onehot = x_onehot.float().transpose(1, 2)  # [batch, 256, seq_len]
        
        x = F.relu(self.conv1(x_onehot))
        
        # Apply intermediate conv layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class CNNRNN(nn.Module):
    def __init__(self, num_classes, max_len, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.max_len = max_len
        self.num_classes = num_classes       
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # First conv layer: vocab_size -> hidden_dim
        self.conv1 = nn.Conv1d(self.vocab_size, self.hidden_dim, kernel_size=3, padding=1)
        
        # GRU layer
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_onehot = F.one_hot(x, num_classes=self.vocab_size)
        x_onehot = x_onehot.float().transpose(1, 2)  # [batch, 256, seq_len]
        
        x = F.relu(self.conv1(x_onehot))
        
        # gru expects (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # Pass through GRU
        _, h_n = self.gru(x)
        
        # Get the last hidden state
        x = h_n[-1, :, :]
        
        return self.classifier(x)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, writer, epoch, metric, logger):
    model.train()
    total_loss = 0
    
    num_samples = len(dataloader.dataset)
    with tqdm(total=num_samples, desc=f"Epoch {epoch+1} Training", unit="sentence") as progress_bar:
        for batch_idx, batch in enumerate(dataloader):
            data, target = batch['input_ids'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=-1)
            metric.add_batch(predictions=predictions.cpu(), references=target.cpu())
            
            # Log batch loss to TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            progress_bar.update(data.shape[0])
            progress_bar.set_postfix(loss=loss.item())

    results = metric.compute()
    accuracy = results['accuracy']
    return total_loss / len(dataloader), accuracy


def byte_level_tokenize(batch, max_length=128):
    # Parallel encoding using list comprehension (automatically optimized)
    encoded_texts = [[b + 1 for b in text.encode('utf-8')][:max_length] for text in batch['sentence']]
    
    # Pad sequences to max_length with 0 (padding token)
    padded_texts = []
    for tokens in encoded_texts:
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        padded_texts.append(tokens)
    
    return {'input_ids': padded_texts, 'label': batch['label']}


def eval(model, dataloader, criterion, device, metric, writer, epoch, wili_langs):
    model.eval()
    total_loss = 0
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            data, target = batch['input_ids'].to(device), batch['label'].to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=-1)
            metric.add_batch(predictions=predictions.cpu(), references=target.cpu())
            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(target.cpu().tolist())

    results = metric.compute()
    accuracy = results['accuracy']

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=wili_langs, yticklabels=wili_langs,
            title='Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)
    
    writer.add_image('Confusion Matrix', image_tensor, epoch)
    plt.close(fig)

    return total_loss / len(dataloader), accuracy


def collate_fn(batch, min_length):
    input_ids, labels = [], []

    for item in batch:
        x = item["input_ids"]
        y = item["label"]

        start = torch.randint(min_length, len(x) + 1, (1,)).item()
        x[start:] = 0  # 0 is the padding token

        input_ids.append(x)
        labels.append(y)

    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.tensor(labels)
    }

def main():
    parser = argparse.ArgumentParser(description='Language Classification Training')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of intermediary CNN layers')
    parser.add_argument('--run_name', type=str, default='language_classification', help='Name of the training run')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--target_lr', type=float, default=1e-5, help='Final learning rate after decay')
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'CNNRNN'], help='Model to use')
    args = parser.parse_args()
    
    # Create run directory
    run_dir = f'runs/{args.run_name}'
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, 'training.log')),
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Run name: {args.run_name}')
    logger.info(f'Hidden dim: {args.hidden_dim}, Num layers: {args.num_layers}')
    logger.info(f'Epochs: {args.num_epochs}, Batch size: {args.batch_size}')
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(run_dir)
    
    logger.info("Loading wili_2018 dataset...")
    ds = load_dataset("MartinThoma/wili_2018")
    
    logger.info("Getting languages from wili_2018...")
    wili_langs = ds['train'].features['label'].names
    lang_to_label = {lang: i for i, lang in enumerate(wili_langs)}
    
    tatoeba_sentences, tatoeba_labels = get_cached_tatoeba_data(logger, wili_langs=wili_langs, lang_to_label=lang_to_label)

    from datasets import Dataset, concatenate_datasets
    tatoeba_ds = Dataset.from_dict({"sentence": tatoeba_sentences, "label": tatoeba_labels}, features=ds['train'].features)
    
    logger.info("Combining datasets...")
    train_ds = concatenate_datasets([ds['train'], tatoeba_ds]).shuffle(seed=42)

    logger.info("Tokenizing to bytes...")
    tokenized_train_ds = train_ds.map(byte_level_tokenize, batched=True, batch_size=1000)
    tokenized_test_ds = ds['test'].map(byte_level_tokenize, batched=True, batch_size=1000)
    
    all_labels = list(ds['train']['label']) + list(ds['test']['label'])
    num_classes = max(all_labels) + 1
    logger.info(f"Number of languages: {num_classes}")
    
    vocab_size = 257
    logger.info(f"Vocabulary size: {vocab_size}")
    
    tokenized_train_ds.set_format(type='torch', columns=['input_ids', 'label'])
    tokenized_test_ds.set_format(type='torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(tokenized_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, min_length=20))
    test_loader = DataLoader(tokenized_test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    if args.model == 'CNN':
        model = CNN(
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                max_len=128,
                vocab_size=vocab_size,
                num_layers=args.num_layers,
        ).to(device)
    elif args.model == 'CNNRNN':
        model = CNNRNN(
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                max_len=128,
                vocab_size=vocab_size,
                num_layers=args.num_layers,
        ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.start_lr, weight_decay=0.01)
    
    def lr_lambda(updates, start_lr, target_lr):
        percentage_remaining = 1 - (updates / (args.num_epochs * len(train_loader)))
        factor = target_lr / start_lr
        return factor + percentage_remaining * ( 1 - factor)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda updates: lr_lambda(updates, args.start_lr, args.target_lr))
    
    # Load accuracy metric once
    accuracy_metric = evaluate.load('accuracy')
    
    # Training loop
    best_test_acc = 0
    
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        logger.info(f'Epoch {epoch + 1}/{args.num_epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, writer, epoch, accuracy_metric, logger)
        test_loss, test_acc = eval(model, test_loader, criterion, device, accuracy_metric, writer, epoch, wili_langs)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Test_Epoch', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_accuracy': test_acc,
                'args': vars(args),
            }, os.path.join(run_dir, 'best_model.pth'))
            logger.info(f'New best model saved with accuracy: {best_test_acc:.4f}')
    
    writer.close()
    logger.info(f'Training completed! Best test accuracy: {best_test_acc:.4f}')
    logger.info(f'TensorBoard logs saved to: {run_dir}')
    logger.info('Run "tensorboard --logdir=runs" to view logs')
    

def get_cached_tatoeba_data(logger, wili_langs, lang_to_label, sample_rate=0.1):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    tatoeba_langs = tatoeba.all_languages
    common_langs = set(wili_langs) & set(tatoeba_langs)
    logger.info(f"Found {len(common_langs)} common languages.")
    
    tatoeba_sentences = []
    tatoeba_labels = []
    
    for lang in common_langs:
        cache_path = cache_dir / f"tatoeba_{lang}_sampled_{sample_rate}.pkl"
        
        if cache_path.exists():
            logger.info(f"Loading cached sampled data for {lang}...")
            with open(cache_path, "rb") as f:
                sentences, labels = pickle.load(f)
        else:
            logger.info(f"Fetching sentences for {lang} from tatoeba...")
            all_sentences = [s.text for s in tatoeba.sentences_detailed(lang)]
            
            num_samples = int(len(all_sentences) * sample_rate)
            sampled_sentences = random.sample(all_sentences, min(num_samples, len(all_sentences)))
            sentences = sampled_sentences
            labels = [lang_to_label[lang]] * len(sentences)
            
            logger.info(f"Sampled {len(sentences)} out of {len(all_sentences)} sentences for {lang}")
            
            # Cache the sampled data
            with open(cache_path, "wb") as f:
                pickle.dump((sentences, labels), f)
        
        tatoeba_sentences.extend(sentences)
        tatoeba_labels.extend(labels)
    
    return tatoeba_sentences, tatoeba_labels

if __name__ == "__main__":
    main()
    