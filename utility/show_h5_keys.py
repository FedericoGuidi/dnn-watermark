import pandas as pd
import sys

def dump_h5_keys(h5fname):
    store = pd.HDFStore(h5fname)
    print('<{}>'.format(h5fname))
    print('\n'.join(store.keys()))
    print()

def hdf5_upd(file_name, file_path, file_data):
    store = pd.HDFStore(file_name)

    if file_path in store.keys():
        store.remove(file_path)
    store.append(file_path, file_data)
    store.close()

if __name__ == '__main__':
    dump_h5_keys(sys.argv[1])
