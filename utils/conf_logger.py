#!/usr/bin/python3.8


import logging
from logging.config import dictConfig


def configure_logger(filename=''):

      logging_config = dict(
            version = 1,
            formatters = {
                  'f': {
                        'format':
                        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
                        }
                  },
            handlers = {
                  'file': {
                        'class': "logging.FileHandler",
                        'filename': "logs/logger.log",
                        'mode': 'w',
                        'formatter': 'f',
                        'level': "DEBUG"
                        },
                  'default': {
                        'formatter': 'f',
                        'class': 'logging.StreamHandler',
                        'stream': 'ext://sys.stdout'
                        }
                  },
            loggers = {
                  '': {
                        'handlers': ['default'],
                        'level': 'WARNING',
                        'propagate': False
                  },
                  'add_binance_vision_zip_to_db.py': {
                        'handlers': ['default'],
                        'level': 'DEBUG',
                        'propagate': False
                  },
                  'database_logger': {
                        'handlers': ['default'],
                        'level': 'INFO',
                        'propagate': False
                  },
                  'preprocessing.py': {
                        'handlers': ['default'],
                        'level': 'INFO',
                        'propagate': False
                  },
                  'generate_predictions.py': {
                        'handlers': ['default'],
                        'level': 'INFO',
                        'propagate': False
                  },
                  "find_and_evaluate_strategy.py": {
                        'handlers': ['default'],
                        'level': 'INFO',
                        'propagate': False
                  }
            }
      )

      dictConfig(logging_config)
      return logging.getLogger(filename)
