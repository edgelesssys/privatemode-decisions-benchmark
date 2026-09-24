"""Privatemode System One against TypeSafe AI's Jev, on labelled data.

Both products expose the same abstraction -- a piece of state, a set of
named options, one typed answer with a probability per option -- so the
comparison can be structural rather than a prompt-engineering contest:
every arm is handed the identical state string, the identical option names
in the identical order, and the identical instruction line.
"""
