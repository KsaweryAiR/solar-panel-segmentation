import json
import onnx

# TO DO: Replace the path with your ONNX model path
model_path = 'path/to/your/model'  # <-- Replace this line with your ONNX model path
model = onnx.load(model_path)

class_names = {
    0: '_background',
    1: 'panel'
}

m1 = model.metadata_props.add()
m1.key = 'model_type'
m1.value = json.dumps('Segmentor')

m2 = model.metadata_props.add()
m2.key = 'class_names'
m2.value = json.dumps(class_names)

m3 = model.metadata_props.add()
m3.key = 'resolution'
m3.value = json.dumps(10)

m4 = model.metadata_props.add()
m4.key = 'standardization_mean'
m4.value = json.dumps([0.485, 0.456, 0.406])

m5 = model.metadata_props.add()
m5.key = 'standardization_std'
m5.value = json.dumps([0.229, 0.224, 0.225])

onnx.save(model, f'{model_name}_with_params.onnx')
