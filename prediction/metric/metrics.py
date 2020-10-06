import numpy as np


class Metrics:

    def __init__(self):
        pass

    def get_metrics_dict(self, predictions, labels):
        res = dict()

        res['rmse'] = self.rmse(predictions, labels)
        res['mae'] = self.mae(predictions, labels)
        res['mape'] = self.mape(predictions, labels)

        return res

    @staticmethod
    def rmse(predictions, labels):
        return np.sqrt(np.mean(np.subtract(predictions, labels) ** 2))

    @staticmethod
    def mae(predictions, labels):
        return np.mean(np.abs(predictions - labels))

    @staticmethod
    def mape(predictions, labels):
        return np.mean(np.abs(np.subtract(predictions, labels) / labels))

    @staticmethod
    def metrics_dict_to_str(metrics_dict):
        eval_info = ''
        for key, value in metrics_dict.items():
            eval_info += '{0} : {1}, '.format(key, value)

        return eval_info[:-1]
