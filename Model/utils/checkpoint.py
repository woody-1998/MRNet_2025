import torch

def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss):
    """
    Saves model state, optimizer state, scheduler state, and current epoch.
    
    Args:
        path (str): File path to save checkpoint (e.g., 'best_model.pth').
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch.
        val_loss (float): Validation loss at the time of saving.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    """
    Loads a previously saved checkpoint into the model (and optionally optimizer and scheduler).
    
    Args:
        path (str): File path of the checkpoint.
        model (torch.nn.Module): Model instance to load the weights into.
        optimizer (torch.optim.Optimizer, optional): If provided, will load its state too.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): If provided, will load its state.
        map_location (str or torch.device): Device to map the checkpoint to (e.g., 'cuda' or 'cpu').

    Returns:
        start_epoch (int): Next epoch to continue training from.
        val_loss (float): Validation loss stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f"Loaded checkpoint from {path} — Epoch: {epoch}, Val Loss: {val_loss:.4f}")
    
    return epoch + 1, val_loss
