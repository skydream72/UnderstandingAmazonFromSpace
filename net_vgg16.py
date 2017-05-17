
import json

from keras.applications.vgg16 import VGG16
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input, Flatten

MODEL_NAME = "VGG16"
# create the base pre-trained model
def build_model(nb_classes):
    base_model = VGG16(weights='imagenet', include_top=False)
    #x = base_model.output
    ## There is still two layers for fine turning, how to do that?
    #x = Flatten(name='flatten', input_shape=base_model.output_shape[1:])(x)
    #x = Dense(4096, activation='relu', name='fc1')(base_model.layers[-4].output)
    #x = Dense(4096, activation='relu', name='fc2')(x)
    #x = Dense(nb_classes, activation='softmax', name='predict_overwrite')(x)
    
    #Create your own input format (here 3x200x200)
    input = Input(shape=(224,224,3),name = 'image_input')

    #Use the generated model 
    output_vgg16_conv =  base_model(input)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    #print("softmax x shape = ", x.get_shape())
    #x = Dense(4096, activation='softmax', name='fc2')(x)
    x = Dense(nb_classes, activation='softmax', name='predict')(x)
  
    model = Model(input=input, output=x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        #print("len base_model.layers = ", layer.name)
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    print "starting model compile"
    compile(model)
    print "model compile done"
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
    # load json and create model
    with open(prefix+"_"+MODEL_NAME+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(prefix+"_"+MODEL_NAME+".h5")
    with open(prefix+"_"+MODEL_NAME+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
