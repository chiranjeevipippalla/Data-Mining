import os
from dominant_attribute import *
from misc import *
from parser import *
from MDLP import MDLP_Discretizer
import sys

if int(sys.version[0]) == 3:
    raw_input = input


def runPipeline(df, method):
    """
    Run processing pipeline

    This function runs the processing pipeline from discretization to output.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset

    Returns
    -------
    pandas.DataFrame
        Data frame representation of the dataset
        :param df:
        :param method:
    """
    attribute_colnames = df.columns[:-1]
    if df[attribute_colnames].select_dtypes(include="number").shape[1] > 0:
        if method == 1:
            print("Running Dominant Attribute Algorithm")
            return dominantAttribute(df)
        else:
            print("Globalized Minimal Conditional Entropy")
            return MCE(df)


def run(input_file, method):
    """
    Run the program

    This function runs the main execution part of the program.
        :param input_file: String
            Path to input file
        :param method: discretization method
    """
    df = parseInput(input_file)
    consistency_level = levelOfConsistency(df)
    is_consistent = consistency_level == 1.0
    base = os.path.splitext(input_file)[0]

    if is_consistent:
        # Data set is consistent
        attr, df = runPipeline(df, method)
        # Print to file
        file_handler_output = open(base + ".int", "w")
        file_handler_output.write(str(attr))
        file_handler_output.close()

        file_handler_output = open(base + ".data", "w")
        file_handler_output.write("[ " + " ".join("{}".format(x) for x in df.columns) + " ]\n")
        file_handler_output.close()
        df.to_csv(base + ".data", sep=" ", index=False, header=False, mode="a")
    else:
        print('Input data set is not consistent')


def MCE(df):
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    df_attributes = df[attribute_colnames]

    # No numerical attribute
    if df_attributes.shape[1] == 0:
        return df

    print("Dominant Attribute: Numerical attributes are {}".format(list(attribute_colnames)))

    numeric_features = np.arange(df_attributes.shape[1])

    # Initialize discretizer object and fit to training data
    discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer.fit(np.asarray(df_attributes), np.asarray(df[decision_colname]))

    cut_points = discretizer.cuts
    for i in range(len(list(attribute_colnames))):
        cut_points[list(attribute_colnames)[i]] = cut_points[i]
        del cut_points[i]
    print("Globalized Minimal Conditional Entropy: cut points = {}".format(cut_points))
    df_discretized = generateDiscretizedDataFrame(df, cut_points)
    df[df_discretized.columns] = df_discretized
    return cut_points, df


def main():
    input_file = raw_input('Enter name of input data file(for example, test.d): ')
    while not os.path.isfile(input_file):
        print('Date file you input does not exist!')
        input_file = raw_input('Enter name of input data file again: ')

    method = int(raw_input('select method(1 for discrimination, 2 for discrimination): '))
    while method != 1 and method != 2:
        print('Wrong input for method!')
        method = int(raw_input('Enter just 1(discrimination) or 2(discrimination): '))

    run(input_file, method)


if __name__ == "__main__":
    main()
