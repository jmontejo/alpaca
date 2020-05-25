import logging

__all__ = ['setup_logger']


def setup_logger(file_path=None):
    root_logger = logging.getLogger()

    # stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    cf = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(cf)
    root_logger.addHandler(ch)

    # file handler
    if file_path is not None:
        fh = logging.FileHandler(str(file_path))
        fh.setLevel(logging.DEBUG)
        ff = logging.Formatter('[%(asctime)s] %(levelname)s :: %(message)s')
        fh.setFormatter(ff)
        root_logger.addHandler(fh)

    root_logger.setLevel(logging.DEBUG)
