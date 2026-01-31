import torch
#Patch torch.load as a backup
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralyticsplus import YOLO, render_result

model = YOLO('mshamrai/yolov8s-visdrone')

#Set model parameters
model.overrides['conf'] = 0.25
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 1000

#Test with image
image = 'person_in_field.png'
results = model.predict(image)

#Display results
print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
render.show()