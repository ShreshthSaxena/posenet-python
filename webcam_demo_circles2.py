import tensorflow as tf
import numpy as np
import cv2
import time
import argparse
import math
import random
import pickle
import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.2)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        with open("listt.txt", "rb") as fp:
            b = pickle.load(fp)
        
        circles = []
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            if pose_scores[0] == 0:
            	continue

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, [pose_scores], keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            k = keypoint_coords[0][-10:]
            k = k.astype(int)

            d = [math.sqrt((k[i][0]-b[i][0])**2 + (k[i][1]-b[i][1])**2) for i in range(-6,0)]
            
            if len(circles)>0:
                #cv2.putText(overlay_image, 'pose matches!!', (20,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                text = "pose matches!!"
                for c in circles:
                    overlay_image = cv2.circle(overlay_image, (c[0],c[1]), 80, (0,150,255), thickness =10)

            elif sum(d)/len(d) < 30:
                #cv2.putText(overlay_image, 'pose matches!!', (20,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                text = "pose matches!!"
                circles = [((b[-2,1]+b[-1,1])//2,min(b[-2,0],b[-1,0])),
                (b[-4,1]+200,b[-4,0]),
                (b[-3,1]-200,b[-3,0])]
            else:
            	for point in b[-6:]:
            		overlay_image = cv2.circle(overlay_image, (point[1],point[0]), 10, (0,150,255), thickness =-1)
            	text = "Move Back"
            	#cv2.putText(overlay_image, "move back", (20,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            overlay_image = cv2.flip(overlay_image,1)
            cv2.putText(overlay_image, text, (20,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

            # if dl<160 or dr<160:
            #     circles[1:] = [((k[-2,1]+k[-1,1])//2,min(k[-2,0],k[-1,0])),
            #     (k[-4,1]+150,k[-4,0]),
            #     (k[-3,1]-150,k[-3,0])]
                # with open("listt.txt", "wb") as fp:
                #     pickle.dump(k, fp)
                
                # for point in k:
                #     temp = cv2.circle(display_image, (point[1],point[0]), 10, (0,150,255), thickness =2)
                # cv2.imwrite('chala.jpg',temp)

            # for c in circles:
            #     if c == circles[0]:
            #         overlay_image = cv2.circle(overlay_image, (c[0],c[1]), 50, (0,0,0), thickness =10)
            #     elif c!= None:
            #         overlay_image = cv2.circle(overlay_image, (c[0],c[1]), 80, (0,150,255), thickness =10)

            # for c in circles:
            # 	dl = math.sqrt((c[0]-xl)**2 + (c[1]-yl)**2)
            # 	dr = math.sqrt((c[0]-xr)**2 + (c[1]-yr)**2)

            # 	if dl<160:
            # 		overlay_image = cv2.circle(overlay_image, (c[0],c[1]), 90, (150,150,0), thickness =10)
            # 	elif dr<160:
            # 		overlay_image = cv2.circle(overlay_image, (c[0],c[1]), 90, (0,150,255), thickness =10)
            # 	else:
            # 		overlay_image = cv2.circle(overlay_image, (c[0],c[1]), 90, (0,0,0), thickness =10)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()