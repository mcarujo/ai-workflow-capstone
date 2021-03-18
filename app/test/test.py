#!/usr/bin/python

import sys
import unittest

from unittests import *
unittest.main()


if __name__ == "__main__":
    """
        Data Processing flow testing.
    """
    test = DataProcessing()
    test.get_dataframe_to_train()

if __name__ == "__main__":
    """
        Model training test.
    """
    from data_processing import DataProcessing
    test1 = DataProcessing()
    test = ModelTrain(test1.get_dataframe_to_train())
    model, metrics = test.run()


    test = ModelPredict()
    print(test.predict(10))
