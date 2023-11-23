import os


def clean(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


