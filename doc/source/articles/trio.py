

import static_frame as sf

# https://www.kaggle.com/unsdsn/world-happiness
# kaggle datasets download -d unsdsn/world-happiness


def main():
    fp = '/tmp/archive.zip'


    sc = sf.StoreConfig(index_depth=1, label_decoder=int, label_encoder=str)

    bus = sf.Bus.from_zip_csv(fp, config=sc)
    f = bus[2017]
    print(f)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()