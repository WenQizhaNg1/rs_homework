from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import cv2 as cv

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file',default='screenshot.png')
    parser.add_argument('--output', help='Output Image file',default='output.png')
    parser.add_argument('--config', help='Config file',default='segformer.b0.512x512.ade.160k.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='latest.pth')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc12',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    img = show_result_pyplot(model, args.img, result, get_palette(args.palette))
    imgpath = "img/"+args.output
    cv.imwrite(imgpath,img)

if __name__ == '__main__':
    main()
