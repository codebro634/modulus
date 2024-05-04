# File created by Robin Schmöcker, Leibniz University Hannover, Germany, Copyright (c) 2024

import json
from os.path import exists


class LogAnalysis:
    def __init__(self, paths):
        self.paths = paths

        #Load log list
        self.original_data = []
        if not isinstance(paths, list):
            paths = [paths]
        for i, path in enumerate(paths):
            if not exists(path):
                raise FileNotFoundError(f"File at path {path} not found")
            with open(path, 'r') as file:
                data = json.load(file)
            for d in data:
                d["n"] = i
                d["path"] = path
            self.original_data += data
        self.data = None

        self.quantity = "v" #v=velocity, p=pressure
        self.error = "all" # 1s, 50s
        self.filter = None


    def get_mse(self, num_outliers=0, get_range:bool = True, take_root: bool = True):
        if self.data is None:
            raise ValueError("No data selected. Use select_data() to select data.")
        errs = []
        for i, d in enumerate(self.data):
            d.sort()
            if num_outliers > 0:
                d = d[num_outliers:-num_outliers]
            errs.append(( sum(d) / len(d)) if not take_root else ( sum(d) / len(d) )**0.5)

        avg_mse = sum(errs) / len(errs)
        return (avg_mse, max(abs(avg_mse-min(errs)), abs(avg_mse-max(errs)))) if get_range else avg_mse

    def print_mses_sorted(self, root_distr: bool = False):
        if self.data is None:
            raise ValueError("No data selected. Use select_data() to select data.")
        for i, d in enumerate(self.data):
            d.sort()
            if root_distr:
                d = [x**0.5 for x in d]
            print(f"Seed {i}: {d}")

    def select_data(self, quantity="v", error="1s", shape_filter=None):
        self.data = [[] for _ in range(len(self.paths))]
        for d in self.original_data:
            if shape_filter is None or shape_filter in d["shapes"]:
                self.data[d["n"]].append(d[quantity][error])


def latex_print(logs,quantity="v", decimal_places=2):

    log_object = LogAnalysis(logs)
    log_object.select_data(quantity=quantity, error="all")
    e_all = log_object.get_mse(take_root=True)
    log_object.select_data(quantity=quantity, error="1s")
    e_1s = log_object.get_mse(take_root=True)
    log_object.select_data(quantity=quantity, error="50s")
    e_50s = log_object.get_mse(take_root=True)
    print(f"${round(e_1s[0],decimal_places)} \pm {round(e_1s[1],decimal_places)}$ & ${round(e_50s[0],decimal_places)}"
          f" \pm {round(e_50s[1],decimal_places)}$ & ${round(e_all[0],decimal_places)} \pm {round(e_all[1],decimal_places)}")

#latex_print(["checkpoints/log_test/details.json","checkpoints/log_test/details.json"])

# test = LogAnalysis(["checkpoints/log_test/details.json","checkpoints/log_test/details.json"])
#
# test.select_data(quantity="v", error="1s")
# print(test.get_mse(take_root=True))
# test.select_data(quantity="v", error="50s")
# print(test.get_mse(take_root=True))
# test.select_data(quantity="p", error="1s")
# print(test.get_mse(take_root=True))
# test.select_data(quantity="p", error="50s")
# print(test.get_mse(take_root=True))
# test.select_data(quantity="v", error="all")
# print(test.get_mse(take_root=True))
# test.select_data(quantity="p", error="all")
# print(test.get_mse(take_root=True))