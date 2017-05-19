
import json

from keras.applications.resnet50 import ResNet50
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input, Flatten

MODEL_NAME = "ResNet50"
# create the base pre-trained model
def build_model(nb_classes):
    base_model = ResNet50(weights='imagenet', include_top=False)

    input = Input(shape=(256,256,3),name = 'image_input')
    #Use the generated model 
    output_resnet50_conv =  base_model(input)
    
    x = Flatten(name='flatten')(output_resnet50_conv)
    #x = GlobalAveragePooling2D()(x)
    #print("GlobalAveragePooling x = " + x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x) #why need extra fully convolution layer?
    # and a logistic layer
    predictions = Dense(nb_classes, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(input=input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def save(model, tags, prefix):
    model.save_weights(prefix+"_"+MODEL_NAME+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open(prefix+"_"+MODEL_NAME+".json", "w") as json_file:
        json_file.write(model_json)
    with open(prefix+"_"+MODEL_NAME+"-labels.json", "w") as json_file:
        json.dump(tags, json_file)


def load(prefix):
    print("open ", prefix+"_"+MODEL_NAME+".json")
    # load json and create model
    with open(prefix+"_"+MODEL_NAME+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    print("open ", prefix+"_"+MODEL_NAME+".h5")
    model.load_weights(prefix+"_"+MODEL_NAME+".h5")
    print("open ", prefix+"_"+MODEL_NAME+"-labels.json")
    with open(prefix+"_"+MODEL_NAME+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

