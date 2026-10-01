"""路径再校验由 L2 prepare_image 负责；读取失败只丢原图，不丢说明。"""
import base64
import logging

from ragagent.context.images import prepare_image
from ragagent.prompt.models import ImagePart

log = logging.getLogger(__name__)


def load_image(image, cfg) -> ImagePart | None:
    try:
        data = prepare_image(image.image_path, cfg)
    except ValueError:
        log.warning('invalid image path; caption only')
        return None
    if data is None:
        return None
    return ImagePart(media_type='image/jpeg', data_b64=base64.b64encode(data).decode('ascii'),
                     chunk_id=image.chunk_id)
