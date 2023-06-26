import os
import sys
import inspect


folder_path = os.path.abspath("torch_activation")
sys.path.append(folder_path)

init_file_path = os.path.join(folder_path, "__init__.py")


class_names = []
module = __import__(folder_path, fromlist=[""])

for name, obj in inspect.getmembers(module):
    if inspect.isclass(obj) and obj.__module__ == "__init__":
        class_names.append(name)

for class_name in class_names:
    print(class_name)
