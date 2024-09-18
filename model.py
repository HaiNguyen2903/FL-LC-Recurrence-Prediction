import torch
import torchvision
import torchvision.transforms as transforms
from IPython import embed
from torchvision.models import mobilenet_v3_large
from torch import nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from dataset import Cancer_Dataset
import wandb
from sklearn.metrics import recall_score, f1_score


class CustomCallback:
    def __init__(self, early_stop_patience=2, reduce_lr_factor=0.2, reduce_lr_patience=3, reduce_lr_min_lr=0.0000001, checkpoint_path='checkpoint.pth', log_dir='logs'):
        # Initialize callback parameters
        self.early_stop_patience = early_stop_patience  # Patience for early stopping
        self.reduce_lr_factor = reduce_lr_factor  # Factor by which to reduce learning rate
        self.reduce_lr_patience = reduce_lr_patience  # Patience for reducing learning rate
        self.reduce_lr_min_lr = reduce_lr_min_lr  # Minimum learning rate
        self.checkpoint_path = checkpoint_path  # Path to save model checkpoints
        # self.log_dir = log_dir  # Directory for logging

        # Initialize variables for early stopping
        self.early_stop_counter = 0  # Counter for early stopping
        self.best_val_loss = float('inf')  # Best validation loss

        self.optimizer = None  # Optimizer for training
        self.scheduler = None  # Learning rate scheduler

    def set_optimizer(self, optimizer):
        # Set optimizer for training
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, val_loss):
        # Early Stopping
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0  # Reset counter if validation loss improves
        else:
            self.early_stop_counter += 1  # Increment counter if validation loss does not improve

        if self.early_stop_counter >= self.early_stop_patience:
            print("Early stopping triggered!")
            return True  # Stop training if early stopping criterion is met

        # Reduce LR on Plateau
        if self.scheduler is not None:
            self.scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        return False  # Continue training

    def on_train_begin(self):
        # Initialize Reduce LR on Plateau scheduler
        self.scheduler = optim.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.reduce_lr_factor,
                                            patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min_lr)

    def on_train_end(self):
        pass

    def set_model(self, model):
        self.model = model  # Set model for the callback

def load_model(num_classes=2):
    model = mobilenet_v3_large(pretrained=True, progress=True)
    
    num_features = model.classifier[0].in_features
    
    model.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Adjusted dropout rate to 0.3 for moderate regularization
            nn.Linear(128, num_classes)  # Directly mapping to the number of classes
        )

    return model
    

def train(model, train_config, train_loader, val_loader):
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (Adam) with weight decay
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    epochs = train_config['epochs']
    batch_size = train_config['train_batch']

    # select device for training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    '''
    start training and evaluating model
    '''

    wandb.watch(model, log_freq=1)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_y_true = []
        train_y_pred = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            labels = batch['recurrence']
            inputs = batch['tumor_img']

            # move to cuda if possible
            if device.type == 'cuda':
                labels = labels.cuda()
                inputs = inputs.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # append to list
            train_y_true.extend(labels.tolist())
            train_y_pred.extend(predicted.tolist())

        
        train_acc = 100 * train_correct / train_total
        train_recall = recall_score(train_y_true, train_y_pred)
        train_f1 = f1_score(train_y_true, train_y_pred)

        print(f'epoch: {epoch+1} \t loss: {round(train_loss, 3)} \t train acc: {round(train_acc, 3)}')
        
        wandb.log({
            'train/loss': train_loss, 
            'train/accuracy': train_acc,
            'train/recall': train_recall,
            'train/f1_score': train_f1
            })

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_y_true = []
        val_y_pred = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                labels = batch['recurrence']
                inputs = batch['tumor_img']

                # move to cuda if possible
                if device.type == 'cuda':
                    labels = labels.cuda()
                    inputs = inputs.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels) 
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # append to list
                val_y_true.extend(labels.tolist())
                val_y_pred.extend(predicted.tolist())

        val_acc = 100 * val_correct / val_total
        val_recall = recall_score(val_y_true, val_y_pred)
        val_f1 = f1_score(val_y_true, val_y_pred)

        print(f'epoch: {epoch+1} \t loss: {round(val_loss, 3)} \t val acc: {round(val_acc, 3)}')
        wandb.log({
            'validation/loss': val_loss, 
            'validation/accuracy': val_acc,
            'validation/recall': val_recall,
            'validation/f1_score': val_f1
            })

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, help='Number of train epochs')
    parser.add_argument('--train_batch', default=64, help='Number of train epochs')

    args = parser.parse_args()

    # init wandb tracking
    wandb.init(project = 'FL-Cancer', tags = ['Federated Learning'])

    train_config = {
        'epochs': args.epochs,
        'train_batch': args.train_batch
    }

    data_root = 'datasets/NSCLC/manifest-1622561851074'

    transform = transforms.Compose([
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cancer_data = Cancer_Dataset(data_root=data_root, 
                          tumor_info_json='data_segmentation_filtered.json',
                          transform=transform)

    train_loader, val_loader, test_loader = cancer_data.get_dataloaders()

    model = load_model()

    train(model, train_config, train_loader, val_loader)

    

    