from ultralytics import YOLO
import cv2
import numpy as np

class Worker:
    def __init__(self, im):
        # cetner of the image
        xc = im.shape[1]//2
        yc = im.shape[0]//2
        self.im = im
        self.im_draw = im.copy()
        s = 100
        self.x1 = xc-s
        self.y1 = yc-s
        self.x2 = xc+s
        self.y2 = yc+s
    def is_inside(self,x,y):
        cv2.rectangle(self.im_draw, (self.x1,self.y1),(self.x2,self.y2), (255,255,0),1)
        cv2.circle(self.im_draw, (x,y), 5, (0,255,0),-1)
        if (self.x1<=x<=self.x2) and (self.y1<=y<=self.y2):
            return True
        else:
            return False

class Agent:
    def __init__(self):
        # Load the model for doing something
        self.model = YOLO("yolo11n-pose.pt") # n,s,m,l,x
    def is_looking_at_the_screen(self, im):
        res = False

        # Prediction
        results = self.model.predict(im, verbose=False)

        # Worker class
        worker = Worker(im)

        # Iterate over the images
        for result in results:
            # There is a person in the image
            for i in range(0, len(result.boxes)):
                x,y,w,h = result.boxes.xyxy.cpu().numpy()[i]
                id = result.boxes.cls.cpu().numpy()[i]
                score = result.boxes.conf.cpu().numpy()[i]
                cv2.putText(im, f"{result.names[id]}: {score}", (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.rectangle(im, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)

                NOSE = result.keypoints.xy.cpu().numpy()[i][0].astype(np.int32)
                LEFT_EYE = result.keypoints.xy.cpu().numpy()[i][1].astype(np.int32)
                RIGHT_EYE = result.keypoints.xy.cpu().numpy()[i][2].astype(np.int32)
                #LEFT_EAR = result.keypoints.xy.cpu().numpy()[i][3].astype(np.int32)
                #RIGHT_EAR = result.keypoints.xy.cpu().numpy()[i][4].astype(np.int32)
                #LEFT_SHOULDER = result.keypoints.xy.cpu().numpy()[i][5].astype(np.int32)
                #RIGHT_SHOULDER = result.keypoints.xy.cpu().numpy()[i][6].astype(np.int32)
                #LEFT_ELBOW = result.keypoints.xy.cpu().numpy()[i][7].astype(np.int32)
                #RIGHT_ELBOW = result.keypoints.xy.cpu().numpy()[i][8].astype(np.int32)
                #LEFT_WRIST = result.keypoints.xy.cpu().numpy()[i][9].astype(np.int32)
                #RIGHT_WRIST = result.keypoints.xy.cpu().numpy()[i][10].astype(np.int32)
                #LEFT_HIP = result.keypoints.xy.cpu()[i].numpy()[11].astype(np.int32)
                #RIGHT_HIP = result.keypoints.xy.cpu()[i].numpy()[12].astype(np.int32)
                #LEFT_KNEE = result.keypoints.xy.cpu()[i].numpy()[13].astype(np.int32)
                #RIGHT_KNEE= result.keypoints.xy.cpu()[i].numpy()[14].astype(np.int32)
                #LEFT_ANKLE = result.keypoints.xy.cpu()[i].numpy()[15].astype(np.int32)
                #RIGHT_ANKLE = result.keypoints.xy.cpu()[i].numpy()[16].astype(np.int32)

                res_NOSE= worker.is_inside(NOSE[0],NOSE[1])
                res_LEFT_EYE = worker.is_inside(LEFT_EYE[0],LEFT_EYE[1])
                res_RIGHT_EYE = worker.is_inside(RIGHT_EYE[0],RIGHT_EYE[1])

                cv2.circle(im, (NOSE[0], NOSE[1]), 5, (0, 0, 255), -1)
                cv2.circle(im, (LEFT_EYE[0], LEFT_EYE[1]), 5, (0, 0, 255), -1)
                cv2.circle(im, (RIGHT_EYE[0], RIGHT_EYE[1]), 5, (0, 0, 255), -1)
                cv2.rectangle(im, (worker.x1, worker.y1), (worker.x2, worker.y2), (255,255,0),1)

                if res_NOSE and res_LEFT_EYE and res_RIGHT_EYE:
                    res = True
                    cv2.putText(im, "OK", (NOSE[0],NOSE[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        return res, im
