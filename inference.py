import time
import cv2
import random
import numpy as np
import onnxruntime as ort

class DocumentLayoutDetection():
    def __init__(self, weight, cuda = True) -> None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weight, providers=providers)
        self.names= [
                'headline',
                'doc',
                'cir_stamp',
                'rec_stamp']

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def translate_input(self, cv2_img, img_shape = (640, 640)):
        names = self.names

        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, new_shape=img_shape , auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape

        outname = [i.name for i in self.session.get_outputs()]
        outname

        inname = [i.name for i in self.session.get_inputs()]
        inname

        inp = {inname[0]:im}
        return inp, outname, [ratio, dwdh, names, colors]
    
    def visualize_detection(self, cv2_img, translate_params, outputs):
        ratio, dwdh, names, colors = translate_params
        ori_images = [cv2_img.copy()]
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = names[cls_id]
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

        return ori_images[0]

    def check_cls_exist(self, cls_id, outputs):
        is_exist = bool(sum([True if int(each[5]) == cls_id else False for each in outputs]))
        return is_exist

    def run_infer(self, cv2_img):
        start_time = time.time()
        inp, outname, translate_params = self.translate_input(cv2_img)
        outputs = self.session.run(outname, inp)[0]
        return outputs, translate_params
        # print("done in:", time.time() - start_time)
