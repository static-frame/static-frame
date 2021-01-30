

import static_frame as sf

# https://www.kaggle.com/unsdsn/world-happiness
#

# Other possibility:
# https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs


def main() -> None:
    fp = '/tmp/archive.zip'


    config = sf.StoreConfig(index_depth=0, label_decoder=int, label_encoder=str)

    bus = sf.Bus.from_zip_csv(fp, config=config)

    def normalize(f: sf.Frame) -> sf.Frame:
        fields = ['country', 'score', 'rank', 'gdp']

        def relabel(label):
            for field in fields:
                if field in label.lower():
                    return field
            return label

        return f.relabel(columns=relabel)[fields].set_index(fields[0], drop=True)

    bus = sf.Batch.from_zip_csv(fp, config=config).apply(normalize).to_bus()

    # selection
    batch = sf.Batch(bus.items())
    quilt = sf.Quilt(bus, axis=0, retain_labels=True)

    import ipdb; ipdb.set_trace()


def table():
    name = 'For n Frame of shape (x, y)'
    columns = (  'Bus', 'Batch', 'Quilt')
    records_items = (
    ('ndim',     (1,      1,      2)),
    ('shape',    ('(n,)', '(n,)', '(xn, y) or (x, yn)'  )),
    ('Iterable', (True,   True,   True)),
    ('Iterator', (False,  True,   False)),
    )

    f = sf.Frame.from_records_items(records_items, columns=columns, name=name)
    # print(f)
    print(f.name)
    print(f.to_rst())

if __name__ == '__main__':
    table()
    main()







    # def format(f: sf.Frame) -> sf.Frame:
    #     field_to_name = {
    #         'country': ('Country', 'Country or region'),
    #         'score': ('Score', 'Happiness.Score', 'Happiness Score'),
    #         'rank': ('Overall rank', 'Happiness.Rank', 'Happiness Rank'),
    #         'gdppc': ('Economy (GDP per Capita)', 'Economy..GDP.per.Capita.', 'GDP per capita'),
    #     }

    #     def get_fields(label):
    #         for field, names in field_to_name.items():
    #             if label in names:
    #                 return field
    #         return label

    #     return f.relabel(columns=get_fields)[list(field_to_name)].set_index(
    #             'country', drop=True)