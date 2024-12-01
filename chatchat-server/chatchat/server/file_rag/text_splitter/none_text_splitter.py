from typing import List

from langchain_text_splitters import TextSplitter

from chatchat.utils import build_logger


logger = build_logger()


class NoneTextSplitter(TextSplitter):
    """
    不对文档进行任何切分处理的切分器.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        try:
            return [text]
        except Exception as e:
            logger.error(f"Error in split_text: {e}")
            raise  # 重新抛出异常以便后续处理
