# apenet/nn/utils.py

def should_print_epoch(epoch, epochs, verbose):
    """Return True if this epoch should be printed."""
    is_first = epoch == 0
    is_multiple = ((epoch + 1) % verbose == 0)
    is_last = epoch == epochs - 1
    return is_first or is_multiple or is_last

def print_epoch_status(epoch, epochs, train_loss, train_acc, val_loss=None, val_acc=None):
    """print epoch training/validation stats."""
    msg = (f"Epoch {epoch+1}/{epochs}: "
           f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
    if val_loss is not None and val_acc is not None:
        msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    print(msg)