




import requests
import tempfile
import static_frame as sf
import pandas as pd

def load_frame():
    url = 'https://raw.githubusercontent.com/datasets/airport-codes/master/data/airport-codes.csv'

    # f = tempfile.TemporaryFile()
    # r = requests.get(url)
    # f.write(r.content)
    # f.seek(0)
    # frame = sf.Frame.from_csv(f)


    tf = tempfile.NamedTemporaryFile(delete=False)
    fp = tf.name
    r = requests.get(url)
    with open(fp, 'wb') as f:
        f.write(r.content)
    frame = sf.Frame.from_csv(fp)

    config = sf.DisplayConfig(display_columns=11, display_rows=13)
    msg = frame.display(config)
    print(msg)



if __name__ == '__main__':
    load_frame()
