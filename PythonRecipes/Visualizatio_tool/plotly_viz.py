import plotly.express as px
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import log
def listFiles(paths=None):
    files = []
    list_paths = []
    for path in paths:
        for p, d, f in os.walk(path):
            list_paths.append(p)
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(p, file))
    return files

def plotly_viz(file_path=None):
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    x = df['Time'].tolist()
    y = [2] * len(x)

    linePlot = px.scatter(x=x, y=y, title=file_name)
    hist = px.histogram(x=x, nbins=50,
                        labels= {'x': "Time_from_start_of_capture"},
                        title=file_name)
    # hist.show()
    linePlot.show()


def plotly_viz_sub(file_List=None, Class=None):
    files_name_list = []
    for file in file_List:
        file_name = os.path.basename(file)
        files_name_list.append(file_name)
    fig = make_subplots(rows=6, cols=1, start_cell="bottom-left", subplot_titles=files_name_list)
    row = 1
    for file in file_List:
        df = pd.read_csv(file)
        x = df['Time'].tolist()
        val = 495
        n_bins = val
        x.append(val)
        fig.add_trace(go.Histogram(x=x, nbinsx=n_bins),
                                   row=row, col=1)
        row += 1
    fig.update_layout(title_text=Class)
    fig.show()


def plotly_viz_sub4(file_List=None, Class=None):
    files_name_list = []
    for file in file_List:
        file_name = os.path.basename(file)
        files_name_list.append(file_name)
    # fig = make_subplots(rows=6, cols=1, start_cell="bottom-left", subplot_titles=files_name_list)
    fig = go.Figure()
    row = 1
    dict_hist = {}
    for i in range(0, len(file_List)):
        df = pd.read_csv(file_List[i])
        x = df['Time'].tolist()
        dict_hist[i] = max(x)
    print(dict_hist)
    # for file in file_List:
    #     df = pd.read_csv(file)
    #     x = df['Time'].tolist()
    #     n_bins = max(x)
    #     print(n_bins)
    #     fig.add_trace(go.Histogram(x=x, nbinsx=200))
    #     row += 1
    # fig.update_layout(title_text=Class)
    # fig.show()


def plotly_viz_sub2(file_List=None, Class=None):
    files_name_list = []
    for file in file_List:
        file_name = os.path.basename(file)
        files_name_list.append(file_name)
    fig = make_subplots(rows=6, cols=1, start_cell="bottom-left", subplot_titles=files_name_list)
    fig = go.Figure()
    row = 1
    for file in file_List:
        df = pd.read_csv(file)
        x = df['Time'].tolist()
        y = [2]*len(x)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',),row=row, col=1)
        row += 1
    fig.update_layout(title_text=Class)
    fig.show()


def plotly_viz_sub3(file_List=None, Class=None):
    files_name_list = []
    for file in file_List:
        file_name = os.path.basename(file)
        files_name_list.append(file_name)
    # fig = make_subplots(rows=6, cols=1, start_cell="bottom-left", subplot_titles=files_name_list)
    fig = go.Figure()
    row = 1
    for file in file_List:
        df = pd.read_csv(file)
        _x = df['Time'].tolist()
        x = [log(i,2) for i in _x]
        y = [row]*len(x)
        file_name = os.path.basename(file)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',name=file_name))
        row += 1
    fig.update_layout(title_text=Class)
    fig.show()


if __name__ == '__main__':
    dict_solana2020a = {
                         # 'audio-chat': r"C:\Users\Owner\projects3\Generalization\study_burstiness\audio-chat",
                        # 'audio-stream': r"C:\Users\Owner\projects3\Generalization\study_burstiness\audio-stream",
                        # 'text-chat': r"C:\Users\Owner\projects3\Generalization\study_burstiness\text-chat",
                        'video-stream': r"C:\Users\Owner\projects3\Generalization\study_burstiness\video-stream"
                        }
    for k, v in dict_solana2020a.items():
        files = listFiles([v])
        print(files)
        plotly_viz_sub(file_List=files, Class=k)

    # plotly_viz_sub(files)
    # for file in files:
    #     #     plotly_viz(file_path=file)