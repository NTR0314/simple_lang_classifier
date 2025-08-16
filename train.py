import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import evaluate


class CNN(nn.Module):
    def __init__(self, num_classes, max_len, hidden_dim, vocab_size):
        super().__init__()
        self.max_len = max_len
        self.num_classes = num_classes       
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Need 3 bytes of receptieve field, because apparently bytes are 3 bytes long in utf-8 if not emojis (ignored)
        # TODO: Filter out emojis
        self.conv1 = nn.Conv1d(self.vocab_size, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_onehot = F.one_hot(x, num_classes=self.vocab_size)
        x_onehot = x_onehot.float().transpose(1, 2)  # [batch, 256, seq_len]
        
        x = F.relu(self.conv1(x_onehot))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch, metric):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        data, target = batch['input_ids'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(output, dim=-1)
        metric.add_batch(predictions=predictions.cpu(), references=target.cpu())
        
        # Log batch loss to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
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


def evaluate(model, dataloader, criterion, device, metric):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            data, target = batch['input_ids'].to(device), batch['label'].to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=-1)
            metric.add_batch(predictions=predictions.cpu(), references=target.cpu())
    
    results = metric.compute()
    accuracy = results['accuracy']
    return total_loss / len(dataloader), accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/language_classification')
    
    print("Loading dataset...")
    ds = load_dataset("MartinThoma/wili_2018")
    
    print("Tokenizing to bytes...")
    tokenized_ds = ds.map(byte_level_tokenize, batched=True, batch_size=1000)
    
    all_labels = list(ds['train']['label']) + list(ds['test']['label'])
    num_classes = max(all_labels) + 1
    print(f"Number of languages: {num_classes}")
    
    vocab_size = 257
    print(f"Vocabulary size: {vocab_size}")
    
    tokenized_ds.set_format(type='torch', columns=['input_ids', 'label'])
    
    batch_size = 32
    train_loader = DataLoader(tokenized_ds['train'], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_ds['test'], batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = CNN(
            hidden_dim=32,
            num_classes=num_classes,
            max_len=128,
            vocab_size=vocab_size,
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Log model parameters count to TensorBoard
    writer.add_scalar('Model/Parameters', sum(p.numel() for p in model.parameters()), 0)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Load accuracy metric once
    accuracy_metric = evaluate.load("accuracy")
    
    # Training loop
    num_epochs = 10
    best_test_acc = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch, accuracy_metric)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, accuracy_metric)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Test_Epoch', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_accuracy': test_acc,
            }, 'best_model.pth')
            print(f'New best model saved with accuracy: {best_test_acc:.4f}')
    
    writer.close()
    print(f'\nTraining completed! Best test accuracy: {best_test_acc:.4f}')
    print(f'TensorBoard logs saved to: runs/language_classification')
    print('Run "tensorboard --logdir=runs" to view logs')

if __name__ == "__main__":
    main()