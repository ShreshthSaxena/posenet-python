from ffpyplayer.player import MediaPlayer
import time
import cv2
player = MediaPlayer('tt.mov')
val = ''

cap = cv2.VideoCapture(0)

# while val != 'eof':
# 	ret, cframe = cap.read()
# 	#_, val = player.get_frame()
# 	cv2.imshow('frame',cframe)
# 	#if val != 'eof' and frame is not None:
# 		#img, t = frame

# cap.release()
# cv2.destroyAllWindows()
# print('hogya bas')



while(val != 'eof'):
    # Capture frame-by-frame
    ret, frame = cap.read()
    _, val = player.get_frame()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
