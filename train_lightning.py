import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from torchmetrics import Accuracy
import argparse
import os
import logging
from tatoebatools import tatoeba
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
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


class LanguageClassificationModule(pl.LightningModule):
    def __init__(self, model_type, num_classes, max_len, hidden_dim, vocab_size, num_layers,
                 start_lr, target_lr, weight_decay, num_training_steps, wili_langs):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        if model_type == 'CNN':
            self.model = CNN(num_classes, max_len, hidden_dim, vocab_size, num_layers)
        elif model_type == 'CNNRNN':
            self.model = CNNRNN(num_classes, max_len, hidden_dim, vocab_size, num_layers)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        # For confusion matrix
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch['input_ids'], batch['label']
        output = self(data)
        loss = self.criterion(output, target)
        
        # Calculate accuracy
        preds = torch.argmax(output, dim=-1)
        acc = self.train_accuracy(preds, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch['input_ids'], batch['label']
        output = self(data)
        loss = self.criterion(output, target)
        
        # Calculate accuracy
        preds = torch.argmax(output, dim=-1)
        acc = self.val_accuracy(preds, target)
        
        # Store predictions for confusion matrix
        self.val_predictions.extend(preds.cpu().tolist())
        self.val_targets.extend(target.cpu().tolist())
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    
    # def on_validation_epoch_end(self):
    #     # Generate and log confusion matrix
    #     if len(self.val_predictions) > 0:
    #         self._log_confusion_matrix(self.val_predictions, self.val_targets, 'val')
    #         # Clear lists for next epoch
    #         self.val_predictions = []
    #         self.val_targets = []
    
    def _log_confusion_matrix(self, predictions, targets, stage):
        cm = confusion_matrix(targets, predictions)
        
        # Create figure
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.hparams.wili_langs, 
               yticklabels=self.hparams.wili_langs,
               title=f'Confusion Matrix ({stage})',
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Log to tensorboard
        if self.logger:
            self.logger.experiment.add_figure(f'confusion_matrix/{stage}', fig, self.current_epoch)
        
        plt.close(fig)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.start_lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Custom learning rate scheduler
        def lr_lambda(current_step):
            percentage_remaining = 1 - (current_step / self.hparams.num_training_steps)
            factor = self.hparams.target_lr / self.hparams.start_lr
            return factor + percentage_remaining * (1 - factor)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


class LanguageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, max_length=128, min_length=20):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.min_length = min_length
        self.wili_langs = None
        self.lang_to_label = None
        
    def prepare_data(self):
        # Download datasets (this is called only on one GPU in distributed setting)
        load_dataset("MartinThoma/wili_2018")
    
    def setup(self, stage=None):
        # Load datasets
        ds = load_dataset("MartinThoma/wili_2018")
        
        # Get language mappings
        self.wili_langs = ds['train'].features['label'].names
        self.lang_to_label = {lang: i for i, lang in enumerate(self.wili_langs)}
        
        # Get Tatoeba data
        tatoeba_sentences, tatoeba_labels = get_cached_tatoeba_data(
            self.wili_langs, self.lang_to_label
        )
        
        # Create Tatoeba dataset
        tatoeba_ds = Dataset.from_dict(
            {"sentence": tatoeba_sentences, "label": tatoeba_labels}, 
            features=ds['train'].features
        )
        
        # Combine datasets
        train_ds = concatenate_datasets([ds['train'], tatoeba_ds]).shuffle(seed=42)
        
        # Tokenize datasets
        self.train_dataset = train_ds.map(
            lambda x: byte_level_tokenize(x, self.max_length), 
            batched=True, 
            batch_size=1000
        )
        self.val_dataset = ds['test'].map(
            lambda x: byte_level_tokenize(x, self.max_length), 
            batched=True, 
            batch_size=1000
        )
        
        # Set format for PyTorch
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'label'])
        self.val_dataset.set_format(type='torch', columns=['input_ids', 'label'])
        
        # Calculate number of classes
        all_labels = list(ds['train']['label']) + list(ds['test']['label'])
        self.num_classes = max(all_labels) + 1
        self.vocab_size = 257
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(x, self.min_length),
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    

def byte_level_tokenize(batch, max_length=128):
    # Parallel encoding using list comprehension
    encoded_texts = [[b + 1 for b in text.encode('utf-8')][:max_length] 
                     for text in batch['sentence']]
    
    # Pad sequences to max_length with 0 (padding token)
    padded_texts = []
    for tokens in encoded_texts:
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        padded_texts.append(tokens)
    
    return {'input_ids': padded_texts, 'label': batch['label']}


def collate_fn(batch, min_length):
    input_ids, labels = [], []

    for item in batch:
        x = item["input_ids"].clone()  # Clone to avoid modifying original
        y = item["label"]

        # Random truncation for data augmentation
        start = torch.randint(min_length, len(x) + 1, (1,)).item()
        x[start:] = 0  # 0 is the padding token

        input_ids.append(x)
        labels.append(y)

    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.tensor(labels) if not isinstance(labels[0], torch.Tensor) else torch.stack(labels)
    }


def get_cached_tatoeba_data(wili_langs, lang_to_label, sample_rate=0.1):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    tatoeba_langs = tatoeba.all_languages
    common_langs = set(wili_langs) & set(tatoeba_langs)
    print(f"Found {len(common_langs)} common languages.")
    
    tatoeba_sentences = []
    tatoeba_labels = []
    
    for lang in tqdm(common_langs, desc="Loading Tatoeba data"):
        cache_path = cache_dir / f"tatoeba_{lang}_sampled_{sample_rate}.pkl"
        
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                sentences, labels = pickle.load(f)
        else:
            all_sentences = [s.text for s in tatoeba.sentences_detailed(lang)]
            
            num_samples = int(len(all_sentences) * sample_rate)
            sampled_sentences = random.sample(all_sentences, min(num_samples, len(all_sentences)))
            sentences = sampled_sentences
            labels = [lang_to_label[lang]] * len(sentences)
            
            # Cache the sampled data
            with open(cache_path, "wb") as f:
                pickle.dump((sentences, labels), f)
        
        tatoeba_sentences.extend(sentences)
        tatoeba_labels.extend(labels)
    
    return tatoeba_sentences, tatoeba_labels


def main():
    parser = argparse.ArgumentParser(description='Language Classification Training with PyTorch Lightning')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'CNNRNN'], 
                       help='Model to use')
    parser.add_argument('--hidden_dim', type=int, default=32, 
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, 
                       help='Number of intermediary CNN layers')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training and evaluation')
    parser.add_argument('--start_lr', type=float, default=1e-3, 
                       help='Initial learning rate')
    parser.add_argument('--target_lr', type=float, default=1e-5, 
                       help='Final learning rate after decay')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                       help='Weight decay for optimizer')
    
    # Other arguments
    parser.add_argument('--run_name', type=str, default='language_classification', 
                       help='Name of the training run')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0, 
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='32', choices=['16', '32', 'bf16'],
                       help='Training precision')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                       help='Accumulate gradients over k batches')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    
    args = parser.parse_args()
    
    # Set up logging
    run_dir = f'runs/{args.run_name}'
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize data module
    print("Initializing data module...")
    data_module = LanguageDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup data to get dimensions
    data_module.setup()
    
    # Calculate total training steps for scheduler
    num_training_steps = (len(data_module.train_dataset) // args.batch_size) * args.num_epochs
    
    # Initialize model
    print("Initializing model...")
    model = LanguageClassificationModule(
        model_type=args.model,
        num_classes=data_module.num_classes,
        max_len=128,
        hidden_dim=args.hidden_dim,
        vocab_size=data_module.vocab_size,
        num_layers=args.num_layers,
        start_lr=args.start_lr,
        target_lr=args.target_lr,
        weight_decay=args.weight_decay,
        num_training_steps=num_training_steps,
        wili_langs=data_module.wili_langs
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename='best-checkpoint-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=5,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir='runs',
        name=args.run_name,
        default_hp_metric=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        logger=tb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=True,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Test model with best checkpoint
    print("Testing best model...")
    trainer.test(model, data_module, ckpt_path='best')
    
    print(f"Training completed! Logs saved to: {run_dir}")
    print('Run "tensorboard --logdir=runs" to view logs')


if __name__ == "__main__":
    main()