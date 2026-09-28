"""
Project: VD-MIL: A Deep Multiple Instance Learning Approach for Violence Detection in Surveillance Videos
Author: Roger Figueroa Quintero
Years: 2025–2026

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

This code is part of an academic/research project.
You are free to use, modify, and share this code for non-commercial purposes only,
provided that proper credit is given to the author.

Commercial use of this code is strictly prohibited without explicit written permission
from the author.

Full license text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""


import os

import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from movinets import MoViNet
from movinets.config import _C

import time


class Net(nn.Module):
      def __init__(self):
         super(Net, self).__init__()
         self.linear_relu_stack = nn.Sequential(
         nn.Linear(2048, 1024),
         nn.ReLU(),
         #nn.Dropout(0.6),
         nn.Linear(1024, 512),
         nn.ReLU(),
         #nn.Dropout(0.6),
         nn.Linear(512, 1),
         #nn.Dropout(0.6),
         nn.Sigmoid(),
         )
         

      def forward(self, x):
          return self.linear_relu_stack(x)

from datetime import datetime

class MovinetClassifier:
      def __init__(self, path_movinet_model, detector_models, device = 'cpu'):
           
          self.device = device
          ####### Movinet backbone #########
          self.model_movinet = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True, model_dir=path_movinet_model)
          self.model_movinet.classifier[3] = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
          self.model_movinet.to(device)
          self.model_movinet.eval()
          ##################################        
          
          ############ Detector ############
          self.classifier = len(detector_models)*[None]
          for i, model_file in enumerate(detector_models):
              self.classifier[i] = torch.load(model_file, map_location=torch.device(device), weights_only=False) 
              self.classifier[i].eval() 
          ##################################
          

      @torch.no_grad()
      def __call__(self, list_clips):
         '''
         Each clip (in this case, each clip is a list of frames) must
         contain T frames:

         list_clips = [[clip_0], [clip_1] ... [clip_n]]

         where  clip_i = [img_0, img_1, ...img_T-1]

         Notes:
         *** The size of each frame does not matter, since they will
             be normalized to 172x172.
         *** All clips must have T frames.

         *** Frames must be provided in BGR format (OpenCV format).
             
         '''
         
         #ti = time.time()

         list_tensor_clips = len(list_clips)*[None]
         for i, clip in enumerate(list_clips):
             ###################################################
             ''' 
             print(clip[0].shape)
             file_name = datetime.utcnow().strftime("%Y%m%d%H%M%S") + ".jpg"
             cv2.imwrite(os.path.join(test_path, file_name), clip[4])
             '''  
             ##################################################

             # Format and resize
             clip_rsz = [cv2.resize(img, (172, 172), cv2.INTER_AREA) for img in clip]# resize to 172x172             
             clip_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in clip_rsz]# convert to RGB 
             list_tensor = [torch.from_numpy(img)[None] for img in clip_rgb]# convert to tensor
            
             
             #list_tensor = [t.to(self.device) for t in list_tensor]

             # Adaptations and normalization
             tensor_clip = torch.cat(list_tensor, axis = 0)
             tensor_clip = tensor_clip.to(self.device)

             tensor_clip = tensor_clip.permute(3,0,1,2)
             tensor_clip = tensor_clip[None]
             tensor_clip = tensor_clip.to(torch.float32)
             tensor_clip = tensor_clip /255.0
             
             list_tensor_clips[i] = tensor_clip
         
         # Create the batch to be processed
         batch = torch.cat(list_tensor_clips, axis = 0)
         #batch = batch.to(self.device)
         #print('batch shape: ', batch.shape)
         ################ Processing #################
         self.model_movinet.clean_activation_buffers()

         xf = self.model_movinet(batch)
         norm_xf = F.normalize(xf, dim=1, p =2)

         # n clips x c categories
         scores_nxc = len(self.classifier)*[None]
         for i, net in enumerate(self.classifier):
             y = net(norm_xf) 
             scores_nx1 = y.to('cpu').numpy()
             scores_nxc[i] = scores_nx1 
         #self.model_movinet.clean_activation_buffers() 
         scores_nxc = np.concatenate(scores_nxc, axis = 1)
         ################################################
         
         #print('Elapsed time: ', time.time() - ti)
         return scores_nxc

 
if __name__== "__main__" :

   import time

   N = 3 # Number of clips
   T = 8 # Temporal length
   #T = 5 # Temporal length

   path_movinet_model = './movinet_weights'
   detector_models = ['./path_models/model_1.pt',
                      './path_models/model_2.pt',
                      './path_models/model_3.pt',
                     ]

   classifier = MovinetClassifier(path_movinet_model, detector_models, device = 'cpu')

   # Simulating a video
   list_clips = []
   for i in range(N):
       clip = []
       for j in range(T):
           black_img = np.zeros((480, 640,3), np.uint8)
           clip.append(black_img)
       list_clips.append(clip)


   for i in range(1000):
       t = time.time()
       scores = classifier(list_clips)
       print(scores)
       print(scores.shape)
       print('elapsed: ', time.time() - t)
       time.sleep(1)
