import pip

module_names = ['os','pandas','joblib','sklearn','FastAPI','pydantic']

for name in module_names:

    try:
        pip.main(['install', name])
        print("Le module "+name+" a été installé")
    except:
        print("Le module "+name+"n'a pas pu être installé")

