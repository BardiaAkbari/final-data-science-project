import os
import sys
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from scipy import stats
from PIL import Image
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


DATASET_PATH = "/kaggle/input/plantvillage-dataset/PlantVillage"
CHECKPOINT_DIR = "./checkpoints_custom_cnn"
RESUME_FROM_CHECKPOINT = None 

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PlantVillageDatasetPreparer:
    def __init__(self, dataset_path: str, data_dir: str = "./data"):
        self.dataset_path = Path(dataset_path)
        self.processed_data_path = Path(data_dir) / "plantvillage_processed"
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    def prepare_data_splits(self):
        print("Preparing data splits...")
        color_path = self.dataset_path
        if (self.dataset_path / "color").exists(): color_path = self.dataset_path / "color"
        elif (self.dataset_path / "PlantVillage").exists(): color_path = self.dataset_path / "PlantVillage"

        class_names = [d.name for d in color_path.iterdir() if d.is_dir()]
        
        train_classes = [
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Apple___Apple_scab',
            'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Peach___Bacterial_spot', 'Peach___healthy', 'Blueberry___healthy'
        ]
        val_classes = [
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Raspberry___healthy'
        ]
        test_classes = [
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy'
        ]

        splits = {'meta_train': train_classes, 'meta_val': val_classes, 'meta_test': test_classes}
        for split_name, classes in splits.items():
            split_path = self.processed_data_path / split_name
            split_path.mkdir(parents=True, exist_ok=True)
            for class_name in classes:
                src_path = color_path / class_name
                dst_path = split_path / class_name
                if src_path.exists() and class_name in class_names:
                    if dst_path.exists(): shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
        print("Data splits prepared successfully!")

class EpisodicDataset(Dataset):
    def __init__(self, data_path: str, n_way: int, k_shot: int, q_query: int, transform=None):
        self.data_path = Path(data_path)
        self.n_way, self.k_shot, self.q_query = n_way, k_shot, q_query
        self.transform = transform
        self.classes = [d.name for d in self.data_path.iterdir() if d.is_dir()]
        self.class_to_images = {c: list(self.data_path.joinpath(c).glob('*.*')) for c in self.classes}
        min_imgs = self.k_shot + self.q_query
        self.classes = [c for c in self.classes if len(self.class_to_images[c]) >= min_imgs]

    def __len__(self):
        return 20000 

    def __getitem__(self, idx):
        episode_classes = random.sample(self.classes, self.n_way)
        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
        for i, c_name in enumerate(episode_classes):
            imgs = random.sample(self.class_to_images[c_name], self.k_shot + self.q_query)
            for img_path in imgs[:self.k_shot]:
                s_imgs.append(self._load_image(img_path))
                s_lbls.append(i)
            for img_path in imgs[self.k_shot:]:
                q_imgs.append(self._load_image(img_path))
                q_lbls.append(i)
        return torch.stack(s_imgs), torch.tensor(s_lbls), torch.stack(q_imgs), torch.tensor(q_lbls)

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image) if self.transform else image


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class CustomCNN(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = output_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CustomCNN()
        self.embed_dim = self.encoder.output_dim
        print(f"Loaded CustomCNN with embedding dimension: {self.embed_dim}")

    def forward(self, support_images, support_labels, query_images, n_way, k_shot):
        all_images = torch.cat([support_images, query_images], dim=0)
        
        embeddings = self.encoder(all_images)
        
        s_embed = embeddings[:support_images.size(0)]
        q_embed = embeddings[support_images.size(0):]
        
        prototypes = torch.stack([s_embed[support_labels == c].mean(0) for c in range(n_way)])
        distances = -torch.cdist(q_embed, prototypes)
        return F.log_softmax(distances, dim=1)


class MetaLearner:
    def __init__(self, model, device, n_way=5, k_shot=5, q_query=5, checkpoint_dir="./checkpoints_custom_cnn"):
        self.model = model.to(device)
        self.device = device
        self.n_way, self.k_shot, self.q_query = n_way, k_shot, q_query
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = defaultdict(list)

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc, 'training_history': self.training_history
        }
        last_model_path = self.checkpoint_dir / "last_model_checkpoint.pth"
        torch.save(checkpoint, last_model_path)
        print(f"Last model checkpoint saved: {last_model_path}")
        if is_best:
            best_model_path = self.checkpoint_dir / "best_model_checkpoint.pth"
            torch.save(checkpoint, best_model_path)
            print(f"🎉 New best model saved: {best_model_path}")
            
    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        print(f"Resuming from epoch {self.start_epoch} with best val acc: {self.best_val_acc:.4f}")

    def _run_episode(self, s_imgs, s_lbls, q_imgs, q_lbls, is_train=True):
        s_imgs, s_lbls = s_imgs.to(self.device), s_lbls.to(self.device)
        q_imgs, q_lbls = q_imgs.to(self.device), q_lbls.to(self.device)
        logits = self.model(s_imgs, s_lbls, q_imgs, self.n_way, self.k_shot)
        loss = self.criterion(logits, q_lbls)
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        acc = (torch.argmax(logits, dim=1) == q_lbls).float().mean()
        return loss.item(), acc.item()

    def meta_train(self, train_loader, val_loader, epochs, episodes_per_epoch, val_episodes):
        print(f"Starting meta-training from epoch {self.start_epoch} to {epochs}...")
        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            train_losses, train_accs = [], []
            with tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch+1}/{epochs} - Training") as pbar:
                for _ in pbar:
                    s_img, s_lbl, q_img, q_lbl = next(iter(train_loader))
                    loss, acc = self._run_episode(s_img.squeeze(0), s_lbl.squeeze(0), q_img.squeeze(0), q_lbl.squeeze(0))
                    train_losses.append(loss); train_accs.append(acc)
                    pbar.set_postfix({'Loss': f'{np.mean(train_losses):.4f}', 'Acc': f'{np.mean(train_accs):.4f}'})
            self.model.eval()
            val_losses, val_accs = [], []
            with torch.no_grad(), tqdm(range(val_episodes), desc=f"Epoch {epoch+1}/{epochs} - Validation") as pbar:
                for _ in pbar:
                    s_img, s_lbl, q_img, q_lbl = next(iter(val_loader))
                    loss, acc = self._run_episode(s_img.squeeze(0), s_lbl.squeeze(0), q_img.squeeze(0), q_lbl.squeeze(0), False)
                    val_losses.append(loss); val_accs.append(acc)
                    pbar.set_postfix({'Loss': f'{np.mean(val_losses):.4f}', 'Acc': f'{np.mean(val_accs):.4f}'})
            
            train_loss_avg, train_acc_avg = np.mean(train_losses), np.mean(train_accs)
            val_loss_avg, val_acc_avg = np.mean(val_losses), np.mean(val_accs)
            
            self.training_history['train_losses'].append(train_loss_avg)
            self.training_history['train_accs'].append(train_acc_avg)
            self.training_history['val_losses'].append(val_loss_avg)
            self.training_history['val_accs'].append(val_acc_avg)
            print(f"\nEpoch {epoch+1} Summary: Train Loss: {train_loss_avg:.4f}, Acc: {train_acc_avg:.4f} | Val Loss: {val_loss_avg:.4f}, Acc: {val_acc_avg:.4f}")
            
            is_best = val_acc_avg > self.best_val_acc
            if is_best: self.best_val_acc = val_acc_avg
            self.save_checkpoint(epoch, val_acc_avg, is_best)
            self.scheduler.step()
            print(f"Learning rate updated to: {self.scheduler.get_last_lr()[0]:.6f}\n")
        print(f"Meta-training completed! Best validation accuracy: {self.best_val_acc:.4f}")
        return self.checkpoint_dir / "best_model_checkpoint.pth"

    def meta_test(self, test_loader, num_episodes, model_path=None):
        print("Starting meta-testing...")
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {model_path}")
        
        self.model.eval()
        test_accs = []
        with torch.no_grad(), tqdm(range(num_episodes), desc="Meta-testing") as pbar:
            for _ in pbar:
                s_img, s_lbl, q_img, q_lbl = next(iter(test_loader))
                _, acc = self._run_episode(s_img.squeeze(0), s_lbl.squeeze(0), q_img.squeeze(0), q_lbl.squeeze(0), False)
                test_accs.append(acc)
                pbar.set_postfix({'Acc': f'{np.mean(test_accs):.4f}'})
        
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        ci = stats.t.interval(0.95, len(test_accs) - 1, loc=mean_acc, scale=stats.sem(test_accs))
        print(f"\nMeta-testing completed! Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"95% Confidence interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
        return mean_acc, ci

def create_transforms():
    im_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.ToTensor(),
        transforms.Normalize(*im_stats)])
    eval_tfm = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(*im_stats)])
    return train_tfm, eval_tfm

def plot_class_distribution(processed_data_path: Path, output_dir: Path):
    split_counts = {}
    splits = ['meta_train', 'meta_val', 'meta_test']
    for split in splits:
        split_path = processed_data_path / split
        if not split_path.exists():
            continue
        
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        counts = {d.name: len(list(d.glob('*.*'))) for d in class_dirs}
        split_counts[split] = counts

    fig, axes = plt.subplots(len(splits), 1, figsize=(15, 8 * len(splits)))
    if len(splits) == 1: axes = [axes] 

    for ax, split in zip(axes, splits):
        if split in split_counts:
            classes = list(split_counts[split].keys())
            counts = list(split_counts[split].values())
            
            bars = ax.bar(classes, counts, color='skyblue')
            ax.set_title(f'Class Distribution for {split} Split', fontsize=16)
            ax.set_ylabel('Number of Images', fontsize=12)
            ax.tick_params(axis='x', labelrotation=90, labelsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, yval, ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle('PlantVillage Dataset Class Distribution', fontsize=20, y=0.99)
    
    plot_path = output_dir / "class_distribution.png"
    plt.savefig(plot_path)
    print(f"Class distribution plot saved to: {plot_path}")

def main():
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    log_file_path = checkpoint_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = Logger(log_file_path)

    print("=" * 80)
    print("Few-Shot Plant Disease Classification with Prototypical Networks (Custom CNN)")
    print("=" * 80)
    print(f"Log file will be saved to: {log_file_path}")
    
    set_seed(42)
    config = {
        'n_way': 5, 'k_shot': 5, 'q_query': 5, 'epochs': 20, 
        'episodes_per_epoch': 100, 'val_episodes': 50, 'test_episodes': 1000,
        'data_dir': './data',
        'checkpoint_dir': CHECKPOINT_DIR
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    preparer = PlantVillageDatasetPreparer(DATASET_PATH, config['data_dir'])
    preparer.prepare_data_splits()

    plot_class_distribution(preparer.processed_data_path, checkpoint_dir)
    
    train_tfm, eval_tfm = create_transforms()
    path = preparer.processed_data_path
    
    train_ds = EpisodicDataset(path / "meta_train", config['n_way'], config['k_shot'], config['q_query'], train_tfm)
    val_ds = EpisodicDataset(path / "meta_val", config['n_way'], config['k_shot'], config['q_query'], eval_tfm)
    test_ds = EpisodicDataset(path / "meta_test", config['n_way'], config['k_shot'], config['q_query'], eval_tfm)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    
    model = PrototypicalNetwork()
    meta_learner = MetaLearner(model, device, **{k:v for k,v in config.items() if k in ['n_way','k_shot','q_query','checkpoint_dir']})
    
    if RESUME_FROM_CHECKPOINT and Path(RESUME_FROM_CHECKPOINT).exists():
        meta_learner.load_checkpoint(RESUME_FROM_CHECKPOINT)
        
    best_model_path = meta_learner.meta_train(train_loader, val_loader, config['epochs'], config['episodes_per_epoch'], config['val_episodes'])
    meta_learner.meta_test(test_loader, config['test_episodes'], best_model_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Resume with 'last_model_checkpoint.pth'.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)