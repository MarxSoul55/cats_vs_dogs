"""Provides interface for functions that act during the training-progress."""


def report(step, total_steps, current_accuracy, current_objective):
    """
    Prints a step-by-step summary of the model's progress.

    # Parameters
        step (int): Current step.
        total_steps (int): Total amount of steps that are being taken.
        current_accuracy (float): Current accuracy to be reported.
        current_objective (float): Current objective to be reported.
    """
    print('Step: {}/{} | Accuracy: {} | Objective: {}'.format(step, total_steps,
                                                              current_accuracy,
                                                              current_objective))
