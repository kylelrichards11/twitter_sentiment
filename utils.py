def print_time(stime, etime, prefix="Finished in "):
    """ Prints time in appropriate units with labels

    Parameters
    ----------
    stime : float - start time in seconds

    etime : float - end time in seconds

    prefix : string, default='Finished in ' - what to print before the time

    Returns
    -------
    None
    """
    diff = etime - stime
    if diff < 100:
        print(f"{prefix}{diff:.2f} seconds")
    elif diff < 6000:
        print(f"{prefix}{diff/60:.2f} minutes")
    else:
        print(f"{prefix}{diff/3600:.2f} hours")

def print_hist(history):
    """ Prints the training history of a neural network in a nice table

    Parameters
    ----------
    history : keras.callbacks.callbacks.History - training history

    Returns
    -------
    None
    """
    val = False
    if 'val_loss' in history.history.keys():
        header = "| Epoch | Train Acc | Val Acc |"
        val = True
    else:
        header = "| Epoch | Train Acc |"
    sep = '-'*len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    num_epochs = len(history.history['loss'])
    for e in range(num_epochs):
        e_spacing = ' '*(6 - len(f"{e+1}"))
        if val:
            print(f"|{e_spacing}{e+1} |     {history.history['acc'][e]:.3f} |   {history.history['val_acc'][e]:.3f} |")
        else:
            print(f"|{e_spacing}{e+1} |     {history.history['acc'][e]:.3f} |")
    print(sep)
    print()

