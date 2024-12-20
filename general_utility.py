import os

def name_from_path(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def invert(y):
    for i in range(len(y)):
        y[i] = - y[i]

    return y

def remove_trailing_letters(s):
    
    end_index = len(s)
    for i in range(len(s) - 1, -1, -1):
        if s[i].isdigit():
            end_index = i + 1
            break

    return s[:end_index]