from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, 'models', 'mobilenet_v2.h5'))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, 'test_images', "modern-bathroom-fixtures.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)