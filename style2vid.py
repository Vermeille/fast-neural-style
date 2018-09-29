import cv2
from queue import Queue
from threading import Thread
import torchvision.transforms as TF
from transformer import Transformer
import argparse


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        self.Q = Queue(maxsize=queueSize)
        self.transforms = TF.Compose([
            TF.ToTensor(),
            NetInputNorm()
        ])

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                grabbed, frame = self.stream.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
                frame = self.transforms(frame)
                if not grabbed:
                    self.stop()
                    return

                self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def ended(self):
        return self.stopped

    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--outsize', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)

    transformer = Transformer()
    transformer.load_state_dict(torch.load(args.path))
    transformer = transformer.eval().to(device)

    out_size = args.outsize.split(',')
    out_size = (int(out_size[0]), int(out_size[1]))

    cap = FileVideoStream(args.input, 4).start()
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, out_size)
    with torch.no_grad():
        i = 0
        while not cap.ended():
            img0 = cap.read()
            img0 = img0.to(device)
            res = transformer(img0)[0].detach().cpu().numpy()
            res *= 255
            res = res.transpose(1, 2, 0).astype('uint8')
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            out.write(res)
            if i % 30 == 0:
                print("{0:2}:{1:2}  ({2})".format(i // (30*60), (i // 30) % 60, i))
            i += 1
    cap.stop()
    out.release()
    print('DONE')
