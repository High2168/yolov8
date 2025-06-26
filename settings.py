from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
ONLINEVIDEO = 'OnlineVideo'
SOURCES_LIST = [IMAGE, VIDEO, ONLINEVIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'test1.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video1': VIDEO_DIR / 'video1.mp4',
    'video2': VIDEO_DIR / 'video2.mp4',
    'video3': VIDEO_DIR / 'video3.mp4',
    'video4': VIDEO_DIR / 'video4.mp4',
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
DETECTIOS_MODEL = MODEL_DIR / 'yolov8s.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'