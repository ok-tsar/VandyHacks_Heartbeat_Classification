import librosa
import librosa.display

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_graph(audio_array, 
               viz_type, 
               sr = 22050, 
               hop_length = 100, 
               out_file = None,
               user = False, 
               dpi = 200):

    # fig = plt.Figure(figsize=(15,10))
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if viz_type == 'chromagram':
        librosa.display.specshow(audio_array, ax=ax)

    if viz_type == 'spectogram':
        librosa.display.specshow(audio_array, sr=sr, y_axis='log', hop_length=hop_length, ax=ax)

    if viz_type == 'mfcc':
        librosa.display.specshow(audio_array, sr=sr, ax=ax)  

    if out_file is not None:
        # Save fig for user
        if user:
            fig.savefig(out_file)

        # Save fig for model
        else:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            fig.savefig(out_file, transparent=True)
    return