import segmentation_models as sm
import argparse
sm.set_framework('tf.keras')
sm.framework()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main(args):
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    CLASSES = ['lane']
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    model_ = model.load_weights(args.model_path)

    tst_image = cv2.imread(args.ip_path)
    tst_image = cv2.resize(tst_image, (256,256))
    tst_image = np.expand_dims(tst_image, axis =0)

    tst_mask = model.predict(tst_image)
    plt.imshow(tst_mask[0])
    plt.imsave(args.op_path, np.array(tst_mask[0]).reshape(256,256), cmap=cm.gray)
    print("Output image successfully saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semantic segmentation")
    parser.add_argument("--model_path", help="path for the saved model")
    parser.add_argument("--ip_path", help="input image path")
    parser.add_argument("--op_path", help="output image path")
    args = parser.parse_args()
    print(args)
    main(args)
