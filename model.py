import torch
import torchvision
import torchvision.transforms as transforms
from IPython import embed
from torchvision.models import mobilenet_v3_large
from torch import nn
import torch.optim as optim
import argparse
import tqdm

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

def train(model, train_config, train_loader, val_loader):
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (Adam) with weight decay
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    # learning rate schedule
    lr_scheduler = optim.ReduceLROnPlateau(optimizer, mode='min', 
                        factor=0.05, patience=1, min_lr=1e-6)

    # other configs
    early_stop_counter = 0
    early_stop_patience = 2
    best_val_loss = float('inf')

    epochs = train_config['epochs']
    batch_size = train_config['train_batch']

    # Solution
    custom_callback = CustomCallback()

    # Solution
    custom_callback.set_optimizer(optimizer)

    # Solution
    # Set the model for the custom callback
    custom_callback.set_model(model)

    # select device for training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to_device(device)

    '''
    start training model
    '''
    for epoch in range(epochs):
        train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, help='Number of train epochs')
    parser.add_argument('--train_batch', default=64, help='Number of train epochs')

    args = parser.parse_args()

    train_config = {
        'epochs': args.epochs,
        'train_batch': args.train_batch
    }

    

    