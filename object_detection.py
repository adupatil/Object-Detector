import torch
from torch.autograd import Variable
import cv2
from ssd import build_ssd
import imageio
from data import BaseTransform, VOC_CLASSES as labelmap

#Defining function

def detect(frame,net,transform):
    height,width = frame.shape[:2]
    #converting to numpy array
    frame_t = transform(frame)[0]
    # converting into tensor using torch
    # converting rbg to grb
    x = torch.from_numpy(frame_t).permute(2,0,1)
    # coverting x into batches using unsqueeze
    # converting x into a torch variable
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    # first w,h (upper corner) next (lower corner)
    scale = torch.Tensor([width,height,width,height])
    # detetctions = [batches,number of classes, number of occurrences,(score,x0,y0,x1,y1)]
    for i in range(detections.size(1)):
        j=0
        while detections[0,i,j,0]>0.6:
            # getting the points
            pt = (detections[0,i,j,1:]*scale).numpy()
            #creating rectangle around detected object
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),2)
            cv2.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            j+=1
    return frame
# Creating the neural network for SSD
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location = lambda storage, loc: storage))

#creating transformations
#transform = BaseTransform(size to feed in nn, scale on which model was trained)
transform = BaseTransform(net.size,(104/256.0, 117/256.0, 123/256.0))

# Detection 
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4',fps=fps)

for i,frame in enumerate(reader):
    frame = detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()
            
            