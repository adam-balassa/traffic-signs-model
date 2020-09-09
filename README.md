# Traffic sign model
A Python model built to recognize traffic signs
## Modules
### Classifier
A Committee of CNNs model consists of 4 CNNs and an MLP that clusters an image into 43 types of traffic signs
### Detector
A Committee of CNNs model consists of 3 CNNs that finds bounding boxes that should contain a traffic sign of any kind
### Traffic sign detector
The model that combines the detector and the classifier by providing appropriately preprocessed images to the models
### Server
A ZeroRPC server that deploys the Traffic sign detector to any ZeroRPC client, by providing an interface with a detect 
method that requires a path to an existing image in any format that open-cv can handle
### Preprocessor
Preprocessing methods that are required for the detector to work properly