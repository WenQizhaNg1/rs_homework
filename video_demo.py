# coding: utf8
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import cv2 as cv

def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='video file')
    parser.add_argument('output', help='video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc12',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # 读取视频
    cap = cv.VideoCapture(args.video)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(fps,width,height)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    path = 'videos/'+args.output
    out = cv.VideoWriter(path,fourcc,fps,(int(width),int(height)))
    while cap.isOpened():
        ret, frame = cap.read()
             # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # gray = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        # cv.imshow('frame', frame)
        #处理图像
        result = inference_segmentor(model,frame)
        output = show_result_pyplot(model, frame, result, get_palette(args.palette))
        out.write(output)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    # test a single image
    # result = inference_segmentor(model, args.img)
    # show the results
    
if __name__ == '__main__':
    main()