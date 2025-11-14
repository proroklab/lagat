import pickle
import pathlib


def load_dataset(funcs, dir_name, args):
    for func in funcs:
        try:
            file_name = func(args)
            path = pathlib.Path(args.dataset_dir, dir_name, file_name)

            with open(path, "rb") as f:
                dataset = pickle.load(f)
            return dataset
        except:
            print(f"Could not find file: {path}, trying legacy file name.")
    raise FileNotFoundError("Could not find any dataset file.")
