import cv2
import numpy as np
import fbrs_predict

def main():

    image_file = 'path-to-image' 
    downscale = 0.4 # performance increases with downscaling image (removes high frequencies)

    image = cv2.imread(image_file)
    image = cv2.resize(image,(int(image.shape[1]*downscale),int(image.shape[0]*downscale)))

    checkpoint = 'resnet34_dh128_sbd' # download a pretrained model from https://github.com/cviss-lab/fbrs_interactive_segmentation to /weights
    engine = fbrs_predict.fbrs_engine(checkpoint)

    x_coord = []
    y_coord = []
    is_pos = []

    def interactive_win(event, u, v, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            x_coord.append(u)
            y_coord.append(v)        
            is_pos.append(1)
            cv2.circle(image2, (u, v), int(5), (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            x_coord.append(u)
            y_coord.append(v)        
            is_pos.append(0)
            cv2.circle(image2, (u, v), int(5), (255, 0, 0), -1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', interactive_win)
    
    image2 = image

    while (1):
        cv2.imshow('image', image2)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 'Esc' Key
            break

        if k == 13:  # 'Enter' Key

            image = cv2.imread(image_file)
            image = cv2.resize(image,(int(image.shape[1]*downscale),int(image.shape[0]*downscale)))

            mask_pred = engine.predict(x_coord, y_coord, is_pos, image, brs_mode='f-BRS-B') # F-BRS Prediction Function

            if len(image.shape) == 3:
                mask = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
            else:
                mask = np.zeros((image.shape[0],image.shape[1]))

            alpha = 0.8
            mask[mask_pred!=0,:] = [0,0,255]
            image[mask_pred!=0,:] = alpha*mask[mask_pred!=0,:] + (1-alpha)*image[mask_pred!=0,:]
            image2 = np.array(image,dtype=np.uint8)

if __name__ == '__main__':
    main()
