# coding: utf-8

from ..consts import AVAILABLE_MODELS
from .consts import MODEL_LABELS_FILE_DICT, PP_SPACE
from .pp_recognizer import PPRecognizer
from .rapid_recognizer import RapidRecognizer

AVAILABLE_MODELS.register_models(MODEL_LABELS_FILE_DICT, space=PP_SPACE)
