import os
from glob import glob
from pathlib import Path
import numpy as np
from lib.inference import load_movenet
from lib.utils import extractFrame
from src.lib.utils import keypointsDataFromImageFiles


def main():
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..')

    movenet = load_movenet()

    extractFrame(glob(root_dir + '/video/pos/*.mp4'), root_dir + '/img/out/pos')

    keypointsDataPos = keypointsDataFromImageFiles(movenet, glob(root_dir + '/img/out/pos/*.jpg'))
    np.save(root_dir + '/out/pdata.npy', keypointsDataPos)

    extractFrame(glob(root_dir + '/video/neg/*.mp4') + glob(root_dir + '/video/neg/*.MOV'), root_dir + '/img/out/neg')

    keypointsDataNeg = keypointsDataFromImageFiles(movenet, glob(root_dir + '/img/out/neg/*.jpg'))
    np.save(root_dir + '/out/ndata.npy', keypointsDataNeg)


if __name__ == '__main__':
    main()
