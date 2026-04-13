import pickle

filepath = './models/model.ted'
my_obj = {'name':'IA', 'version':'0.0.0b'}

def writing(filepath, obj):
    file_obj = open(filepath, "wb")
    pickle.dump(obj, file_obj)

def reading(filepath):
    file_obj = open(filepath, "rb")
    obj = pickle.load(file_obj)
    print(obj)

writing(filepath, my_obj)