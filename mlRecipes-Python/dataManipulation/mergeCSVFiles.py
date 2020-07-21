import shutil
import subprocess as sb
import os


def listFiles(paths=None):
    files = []
    for path in paths:
        for r, d, f in os.walk(path):
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(r, file))
    return files


def merge_csv_file(source_file_list=None,
                   output_file=None,
                   skip_first_line=True):
    """
        Split file data based on class label and created a new file for each class

                Parameters:
                        :param skip_first_line:
                        :param source_file_list: a list of files to be merged
                        :param output_file: a file with merged data

                Returns:
                        nothing
                        Saves merged file @output_file

    """

    with open(output_file, 'wb') as dst:
        for f in source_file_list:
            with open(f, 'rb') as src:
                if skip_first_line:
                    next(src)
                shutil.copyfileobj(src, dst)


if __name__ == '__main__':
    files = listFiles(paths =[r"C:\Users\Owner\mltat\data\mltat\TestFiles\task-36e\results-ks-test\ks-test-univariate-intra-class"])
    print("Total {} files merged".format(len(files)))
    merge_csv_file(source_file_list=files,
                   output_file=r'C:\Users\Owner\mltat\data\mltat\TestFiles\task-36e\results-ks-test\ks-test-univariate-intra-class\merged_task-36e.csv')