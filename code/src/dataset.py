from anomalib.data import MVTecAD

def get_dataloaders(root: str, category: str = "bottle", train_batch_size: int = 8, eval_batch_size: int = 8):
    """
    Initializes and sets up MVTecAD dataset loaders via Anomalib.
    """
    data = MVTecAD(
        root=root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size
    )
    data.setup()
    return data.train_dataloader(), data.test_dataloader()