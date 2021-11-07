import json



# import numpy as np

class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, *args, **kwargs)
        # print(self.parse_array)

        def post(func):
            def wrapper(*args, **kwargs):
                return tuple(func(*args, **kwargs))
            return wrapper

        self.parse_array = post(self.parse_array)

if __name__ == '__main__':
    # from static_frame.test.test_case import temp_file
    pass

    # with temp_file('.json') as fp:
    #     with open(fp, 'w') as f:
    #         obj = {'a':[3, 4, 5], 'b':None}
    #         json.dump(fp, obj)
    #     with open(fp, 'r') as f:
    #         post = json.load(f)
            # print(post)
            # import ipdb; ipdb.set_trace()