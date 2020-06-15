from sagemaker.tensorflow.serving import Predictor
from tensorflow.python.keras.preprocessing.image import load_img

!wget -O /tmp/test.jpg https://YOURPath.jpeg
file_name = '/tmp/test.jpg'

# test image
from IPython.display import Image
Image(file_name)

# Resize as model was trained after resizing
test_image = load_img(file_name, target_size=(150, 150))
test_image_array = np.array(test_image).reshape((1, 150, 150, 3)).tolist()

# Predict
predictor = Predictor(endpoint_name = "my-endpointname")
print(predictor.predict({"instances": [{"inputs": test_image_array}]}))
