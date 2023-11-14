import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os

def draw_on_image(img, faces):
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int64)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(np.int64)
            #print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                            2)
        if face.gender is not None and face.age is not None:
            cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        for key, value in face.items():
           if key.startswith('landmark_3d'):
            #    print(key, value.shape)
            #    print(value[0:10,:])
               lmk = np.round(value).astype(np.int64)
               for l in range(lmk.shape[0]):
                   color = (255, 0, 0)
                   cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
                              2)
    return dimg

def main():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread(os.path.join(current_file_path, "t1.jpg"))
    faces = app.get(img)
    rimg = draw_on_image(img, faces)
    output_path = os.path.join(current_file_path, "data", "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(os.path.join(output_path, "t1_output.jpg"), rimg)

if __name__ == "__main__":
    main()
