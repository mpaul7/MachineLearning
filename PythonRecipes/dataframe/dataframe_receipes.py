import pandas as pd

def dataframe_receipes():
    ## drop a column
    X = df.drop('column_name', axis=1).copy()
    ## OR
    X = df.iloc[:, :-1]

    ## Copy one column to another dataframe
    y = df['column_name'].copy()

    ## get index value based on a condition
    y_not_zero_index = y > 0

    ## how to replace each value based on index value
    y[y_not_zero_index] = 1

    ## how to plot from DataFrame
    df = pd.DataFrame(data, columns=['c1', 'c2', 'c3'])
    df.plot(x='c1',
            y='c2',
            yerr='std',
            marker='o',
            linestyle='--')
    ## drop a columns in DataFrame
    X = df.drop('column_name', axis=1).copy()


if __name__ == '__main__':
    dataframe_receipes()